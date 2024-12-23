// 引入所需模块和库
// 从 models::with_tracing 模块引入 linear, linear_no_bias, Linear, RmsNorm 等函数和结构
use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
// 引入 candle 库中的类型和方法，用于处理张量和设备
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
// 引入 candle_nn 库中的 Activation 和 VarBuilder，用于指定激活函数和构建模型变量
use candle_nn::{Activation, VarBuilder};
// 引入标准库的 Arc 类型，用于共享内存
use std::collections::VecDeque;
use std::sync::Arc;

// 定义 Config 结构体
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    // vocab_size：词汇表的大小，定义了模型能够识别的唯一词的总数
    pub vocab_size: usize,
    // hidden_size：隐藏层的维度，即每层中每个单元的大小
    pub hidden_size: usize,
    // intermediate_size：前馈网络的中间层大小，一般比 hidden_size 大
    pub intermediate_size: usize,
    // num_hidden_layers：模型中堆叠的隐藏层数量，即 transformer 块的层数
    pub num_hidden_layers: usize,
    // num_attention_heads：自注意力机制的头数，用于多头注意力
    pub num_attention_heads: usize,
    // num_key_value_heads：键值对头的数量，支持使用不同的键和值头
    pub num_key_value_heads: usize,
    // max_position_embeddings：最大位置嵌入的数量，定义模型能够处理的最大序列长度
    pub max_position_embeddings: usize,
    // sliding_window：滑动窗口大小，限制注意力机制的上下文范围
    pub sliding_window: usize,
    // max_window_layers：支持滑动窗口的最大层数
    pub max_window_layers: usize,
    // tie_word_embeddings：是否共享词嵌入权重，如果为 true，输入和输出的词嵌入矩阵会共享
    pub tie_word_embeddings: bool,
    // rope_theta：旋转位置嵌入的频率控制参数
    pub rope_theta: f64,
    // rms_norm_eps：RMS 归一化层中的 epsilon 值，用于避免数值不稳定
    pub rms_norm_eps: f64,
    // use_sliding_window：是否启用滑动窗口机制
    pub use_sliding_window: bool,
    // hidden_act：隐藏层的激活函数类型，定义激活函数，如 ReLU、GELU 等
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor, // 存储正弦值的张量，用于旋转嵌入
    cos: Tensor, // 存储余弦值的张量，用于旋转嵌入
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        // 计算单个头的嵌入维度
        let dim = cfg.hidden_size / cfg.num_attention_heads;

        // 设置序列的最大长度
        let max_seq_len = cfg.max_position_embeddings;

        // 计算倒数频率，定义为1/θ^(i/dim)，其中θ由rope_theta控制
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2) // 每隔两个索引（因为sin/cos对一对一组合）
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect(); // 存储所有计算的倒频率

        let inv_freq_len = inv_freq.len(); // 倒频率向量长度
                                           // 创建倒频率张量，并转换为指定的数据类型
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;

        // 创建从0到max_seq_len的张量t，并转换数据类型和形状
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;

        // 通过矩阵乘法计算频率矩阵
        let freqs = t.matmul(&inv_freq)?;

        // 返回包含sin和cos嵌入的RotaryEmbedding实例
        Ok(Self {
            sin: freqs.sin()?, // 计算频率矩阵的正弦
            cos: freqs.cos()?, // 计算频率矩阵的余弦
        })
    }
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,           // 查询张量
        k: &Tensor,           // 键张量
        seqlen_offset: usize, // 序列偏移，用于提取嵌入
    ) -> Result<(Tensor, Tensor)> {
        // 获取查询张量的尺寸信息，提取序列长度
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;

        // 从cos和sin张量中获取适当的切片，长度为当前序列长度
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        // 对查询和键分别应用旋转嵌入
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;

        // 返回应用嵌入后的查询和键张量
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,  // 投影层，用于输入门控机制
    up_proj: Linear,    // 上升投影层，用于提升维度
    down_proj: Linear,  // 降低投影层，将中间结果降回到输入维度
    act_fn: Activation, // 激活函数
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size; // 隐藏层大小
        let intermediate_sz = cfg.intermediate_size; // 中间层大小

        // 创建 gate_proj、up_proj 和 down_proj 三个线性层，无偏置
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;

        // 返回包含各层和激活函数的 MLP 结构体实例
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}
impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 计算 lhs，应用 gate_proj 和激活函数
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;

        // 计算 rhs，只应用 up_proj 投影层
        let rhs = xs.apply(&self.up_proj)?;

        // 将 lhs 和 rhs 相乘，并应用 down_proj 得到最终输出
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
#[repr(align(64))]
struct H2OCache {
    hh_k: VecDeque<Tensor>,       // Heavy Hitters的K缓存
    hh_v: VecDeque<Tensor>,       // Heavy Hitters的V缓存
    recent_k: VecDeque<Tensor>,   // Recent的K缓存
    recent_v: VecDeque<Tensor>,   // Recent的V缓存
    hh_scores: VecDeque<f64>,     // Heavy Hitters的累计attention分数
    recent_scores: VecDeque<f64>, // Recent的累计attention分数
    hh_capacity: usize,           // Heavy Hitters容量
    recent_capacity: usize,       // Recent容量
    num_heads: usize,             // 多头注意力的头数
    head_dim: usize,              // 每个头的维度
}

impl H2OCache {
    // !
    fn print_memory_layout(&self) {
        // 获取结构体信息
        let start_addr = self as *const H2OCache;
        let size = std::mem::size_of::<H2OCache>();
        let end_addr = unsafe { (start_addr as *const u8).add(size) };

        println!("\n=== H2OCache 详细内存布局 ===");
        println!("整体信息:");
        println!("起始地址: {:p}", start_addr);
        println!("结尾地址: {:p}", end_addr);
        println!("总大小: {} 字节", size);
        println!("对齐方式: {} 字节", std::mem::align_of::<H2OCache>());

        println!("\nVecDeque 缓存信息:");
        println!(
            "hh_k -> 容量: {}, 长度: {}",
            self.hh_k.capacity(),
            self.hh_k.len()
        );
        println!(
            "hh_v -> 容量: {}, 长度: {}",
            self.hh_v.capacity(),
            self.hh_v.len()
        );
        println!(
            "recent_k -> 容量: {}, 长度: {}",
            self.recent_k.capacity(),
            self.recent_k.len()
        );
        println!(
            "recent_v -> 容量: {}, 长度: {}",
            self.recent_v.capacity(),
            self.recent_v.len()
        );

        println!("\n分数缓存信息:");
        println!(
            "hh_scores -> 容量: {}, 长度: {}",
            self.hh_scores.capacity(),
            self.hh_scores.len()
        );
        println!(
            "recent_scores -> 容量: {}, 长度: {}",
            self.recent_scores.capacity(),
            self.recent_scores.len()
        );

        println!("\n配置参数:");
        println!("Heavy Hitters容量: {}", self.hh_capacity);
        println!("Recent容量: {}", self.recent_capacity);
        println!("注意力头数: {}", self.num_heads);

        // 打印每个字段的偏移量
        println!("\n字段偏移量:");
        println!("hh_k offset: {}", memoffset::offset_of!(H2OCache, hh_k));
        println!("hh_v offset: {}", memoffset::offset_of!(H2OCache, hh_v));
        println!(
            "recent_k offset: {}",
            memoffset::offset_of!(H2OCache, recent_k)
        );
        println!(
            "recent_v offset: {}",
            memoffset::offset_of!(H2OCache, recent_v)
        );
        println!(
            "hh_scores offset: {}",
            memoffset::offset_of!(H2OCache, hh_scores)
        );
        println!(
            "recent_scores offset: {}",
            memoffset::offset_of!(H2OCache, recent_scores)
        );
        println!(
            "hh_capacity offset: {}",
            memoffset::offset_of!(H2OCache, hh_capacity)
        );
        println!(
            "recent_capacity offset: {}",
            memoffset::offset_of!(H2OCache, recent_capacity)
        );
        println!(
            "num_heads offset: {}",
            memoffset::offset_of!(H2OCache, num_heads)
        );
    }

    // !

    // !有处理空间
    fn new(total_capacity: usize, num_heads1: usize, head_dim1: usize) -> Self {
        let hh_capacity = total_capacity / 7;
        let recent_capacity = 6 * total_capacity / 7;
        // println!("\nhh:{} | R{}\n", hh_capacity, recent_capacity);
        Self {
            hh_k: VecDeque::with_capacity(hh_capacity),
            hh_v: VecDeque::with_capacity(hh_capacity),
            recent_k: VecDeque::with_capacity(recent_capacity),
            recent_v: VecDeque::with_capacity(recent_capacity),
            hh_scores: VecDeque::with_capacity(hh_capacity),
            recent_scores: VecDeque::with_capacity(recent_capacity),
            hh_capacity,
            recent_capacity,
            num_heads: num_heads1, // 多头注意力的头数
            head_dim: head_dim1,   // 每个头的维度
        }
    }

    // ! query是新来的
    fn update_attention_for_new_kv(
        &mut self,
        query: &Tensor, // 新token的query
    ) -> Result<()> {
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        //println!("进入了更新分数的函数");
        // 1. 计算heavy hitters的scores并更新
        if !self.hh_k.is_empty() {
            let decay = 0.99; // HH的衰减因子
            for (_idx, (cached_k, score)) in
                self.hh_k.iter().zip(self.hh_scores.iter_mut()).enumerate()
            {
                // 1.1 计算当前key的attention score
                let attn_weights = (query.matmul(&cached_k.transpose(2, 3)?)? * scale)?;
                let attn_probs = candle_nn::ops::softmax(&attn_weights, 1)?;
                // 1.2 计算多头平均
                let sum_result = attn_probs.sum_all()?;
                // 修改这里：先转换为f32，再转为f64
                let scalar_result = sum_result.to_scalar::<f32>()? as f64;
                let avg_score = scalar_result / self.num_heads as f64;

                // 1.3 更新HH分数（带衰减）
                *score = *score * decay + avg_score;
                //println!("idx:{} 的 HH score: {}", idx, *score);
            }
        }

        // 2. 计算recent tokens的scores并更新
        if !self.recent_k.is_empty() {
            //println!("进入recent分数更新");
            for (_idx, (cached_k, score)) in self
                .recent_k
                .iter()
                .zip(self.recent_scores.iter_mut())
                .enumerate()
            {
                //println!("进入recent计算循环");
                // 2.1 计算当前key的attention score
                let attn_weights = (query.matmul(&cached_k.transpose(2, 3)?)? * scale)?;

                let attn_probs = candle_nn::ops::softmax(&attn_weights, 1)?;

                // 2.2 计算多头平均
                // 分步检查计算过程
                //println!("检查 attn_probs shape: {:?}", attn_probs.shape());

                // 分步执行sum_all和类型转换
                //println!("检查 attn_probs shape: {:?}", attn_probs.shape());

                let sum_result = attn_probs.sum_all()?;
                //println!("sum完成: {:?}", sum_result.shape());

                // 修改这里：先转换为f32，再转为f64
                let scalar_result = sum_result.to_scalar::<f32>()? as f64;
                //println!("to_scalar完成: {}", scalar_result);

                let avg_score = scalar_result / self.num_heads as f64;

                //println!("新的分数算出来了！是{}", avg_score);
                // 2.3 更新Recent分数（直接累加）
                *score += avg_score;
                //println!("idx:{} 的 Recent score: {}", idx, *score);
            }
        }

        Ok(())
    }

    // 插入新的KV对
    // !还要改，插入更新的逻辑
    fn insert(&mut self, query: &Tensor, k: Tensor, v: Tensor) -> Result<()> {
        let _ = self.update_attention_for_new_kv(&query);

        if self.recent_k.len() < self.recent_capacity {
            // Recent未满，直接线性添加
            self.recent_k.push_back(k);
            self.recent_v.push_back(v);
            self.recent_scores.push_back(0.0); // 新token的初始分数为0
        } else {
            // Recent满了，需要移除最老的token
            let (old_k, old_v, old_score) = {
                // 同步弹出token和其累积分数
                let k = self.recent_k.pop_front().unwrap();
                let v = self.recent_v.pop_front().unwrap();
                let score = self.recent_scores.pop_front().unwrap();
                (k, v, score)
            };

            // 添加新token到Recent末尾
            self.recent_k.push_back(k);
            self.recent_v.push_back(v);
            self.recent_scores.push_back(0.0); // 新token的初始分数为0

            // 处理老token到HH的迁移
            if self.hh_k.len() < self.hh_capacity {
                // HH未满，线性添加
                self.hh_k.push_back(old_k);
                self.hh_v.push_back(old_v);
                self.hh_scores.push_back(old_score); // 保持原有累积分数
            } else {
                // HH已满，需要比较分数
                if let Some(min_idx) = self.find_min_score_index() {
                    if old_score > self.hh_scores[min_idx] {
                        // 只有当老token的分数大于HH中最小分数时才替换
                        self.hh_k[min_idx] = old_k;
                        self.hh_v[min_idx] = old_v;
                        self.hh_scores[min_idx] = old_score;
                    }
                    // 如果老token分数较小，则直接丢弃
                }
            }
        }
        Ok(())
    }
    // 查找HH中分数最低的索引
    fn find_min_score_index(&self) -> Option<usize> {
        self.hh_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }

    // 获取所有缓存的KV
    fn get_all_kv(&self) -> Result<(Tensor, Tensor)> {
        let total_len = self.hh_k.len() + self.recent_k.len();

        // 预分配容量
        let mut all_k: Vec<&Tensor> = Vec::with_capacity(total_len);
        let mut all_v: Vec<&Tensor> = Vec::with_capacity(total_len);

        // 先添加HH的tokens (保持顺序一致性)
        all_k.extend(self.hh_k.iter());
        all_v.extend(self.hh_v.iter());

        // 再添加Recent的tokens
        all_k.extend(self.recent_k.iter());
        all_v.extend(self.recent_v.iter());

        // 在维度2上拼接 (与原attention实现保持一致)
        let key_states = Tensor::cat(&all_k, 2)?;
        let value_states = Tensor::cat(&all_v, 2)?;

        Ok((key_states, value_states))
    }

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fn printRecentK(&self) {
        let ptr1 = self.recent_k.as_slices().0.as_ptr();
        let len = self.recent_k.len();
        let cap = self.recent_k.capacity();

        // 计算结尾指针 = 起始指针 + 长度
        let end_ptr = unsafe { ptr1.add(len) };

        println!("VecDeque recent_k内存信息:");
        println!("起始地址: {:p}", ptr1);
        println!("结尾地址: {:p}", end_ptr);
        println!("长度: {}", len);
        println!("容量: {}", cap);
    }
    fn printRecentV(&self) {
        // !!!!!!!!!!!!!!!
        let ptr1 = self.recent_v.as_slices().0.as_ptr();

        let len = self.recent_v.len();
        let cap = self.recent_v.capacity();

        // 计算结尾指针 = 起始指针 + 长度
        let end_ptr = unsafe { ptr1.add(len) };

        println!("VecDeque recent_v内存信息:");
        println!("起始地址: {:p}", ptr1);
        println!("结尾地址: {:p}", end_ptr);
        println!("长度: {}", len);
        println!("容量: {}", cap);
        // !!!!!!!!!!!!!!!!!!
    }
    fn printHHK(&self) {
        // !!!!!!!!!!!!!!!
        let ptr1 = self.hh_k.as_slices().0.as_ptr();

        let len = self.hh_k.len();
        let cap = self.hh_k.capacity();

        // 计算结尾指针 = 起始指针 + 长度
        let end_ptr = unsafe { ptr1.add(len) };

        println!("VecDeque hh_k内存信息:");
        println!("起始地址: {:p}", ptr1);
        println!("结尾地址: {:p}", end_ptr);
        println!("长度: {}", len);
        println!("容量: {}", cap);
        // !!!!!!!!!!!!!!!!!!
    }
    fn printHHV(&self) {
        // !!!!!!!!!!!!!!!
        let ptr1 = self.hh_v.as_slices().0.as_ptr();

        let len = self.hh_v.len();
        let cap = self.hh_v.capacity();

        // 计算结尾指针 = 起始指针 + 长度
        let end_ptr = unsafe { ptr1.add(len) };

        println!("VecDeque hh_v内存信息:");
        println!("起始地址: {:p}", ptr1);
        println!("结尾地址: {:p}", end_ptr);
        println!("长度: {}", len);
        println!("容量: {}", cap);
        // !!!!!!!!!!!!!!!!!!
    }
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,                   // 查询投影层
    k_proj: Linear,                   // 键投影层
    v_proj: Linear,                   // 值投影层
    o_proj: Linear,                   // 输出投影层
    num_heads: usize,                 // 多头注意力的头数
    num_kv_heads: usize,              // 键值的头数
    num_kv_groups: usize,             // 键值的组数
    head_dim: usize,                  // 每个头的维度
    hidden_size: usize,               // 隐藏层维度
    rotary_emb: Arc<RotaryEmbedding>, // 旋转嵌入，用于位置编码
    kv_cache: Option<H2OCache>,       // 替换为H2OCache
}

impl Attention {
    // 构造函数，用于创建一个新的 Attention 实例
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 隐藏层的大小，表示特征维度
        let hidden_sz = cfg.hidden_size;
        // 注意力头的数量
        let num_heads = cfg.num_attention_heads;
        // 键值头的数量（用于多头注意力机制）
        let num_kv_heads = cfg.num_key_value_heads;
        // 每组的键值头数量
        let num_kv_groups = num_heads / num_kv_heads;
        // 每个注意力头的维度
        let head_dim = hidden_sz / num_heads;

        // 为查询、键、值和输出分别创建投影层
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        // 返回初始化的 Attention 实例
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None, // 初始情况下，键值缓存为空
        })
    }

    // 前向传播函数，计算注意力输出
    fn forward(
        &mut self,
        xs: &Tensor,                     // 输入张量
        attention_mask: Option<&Tensor>, // 可选的注意力掩码
        seqlen_offset: usize,            // 序列偏移量，用于旋转嵌入
        // ! 缓存总容量
        total_capacity: usize,
    ) -> Result<Tensor> {
        // 获取输入张量的维度，b_sz 为批次大小，q_len 为查询序列长度
        let (b_sz, q_len, _) = xs.dims3()?;

        // 应用查询、键、值的投影层得到投影后的张量
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        // 将查询、键和值的张量重塑为多头注意力结构，并进行维度交换
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 应用旋转嵌入于查询和键张量
        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // 处理KV cache
        let (key_states, value_states) = match &mut self.kv_cache {
            None => {
                // 首次使用，创建新的H2OCache
                // ! 设置适当的容量
                let mut cache = H2OCache::new(total_capacity, self.num_heads, self.head_dim);
                cache.insert(&query_states, key_states.clone(), value_states.clone())?;
                // !!!!!!!!!!!!!!!!!
                // cache.printRecentK();
                // cache.printRecentV();
                // cache.printHHK();
                // cache.printHHV();
                // println!("\n************************************");
                // cache.print_memory_layout();
                // println!("\n************************************");
                // !!!!!!!!!!!!!!!!!
                self.kv_cache = Some(cache);

                (key_states, value_states)
            }
            Some(cache) => {
                // 已有cache，插入新的KV
                cache.insert(&query_states, key_states, value_states)?;
                // !!!!!!!!!!!!!!!!!
                // cache.printRecentK();
                // cache.printRecentV();
                // cache.printHHK();
                // cache.printHHV();
                // println!("\n************************************");
                // cache.print_memory_layout();
                // println!("\n************************************");
                // !!!!!!!!!!!!!!!!!
                cache.get_all_kv()?
            }
        };

        // 将键和值重复，以适应多头注意力中的键值头组结构
        // ! 此处使用的k和v，永远是包含了历史上全部的k和v、拼接后的k和v。而不是最新产生的k和v
        // ! 因此，需要修改，每次只算新来的k和v
        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // ! 打印 KV Cache 更新后的维度
        // println!(
        //     "After KV Cache update - key_states shape: {:?}, value_states shape: {:?}",
        //     key_states.dims(),
        //     value_states.dims()
        // );

        // 调用封装的计算注意力输出函数
        let attn_output = self.compute_attention_scores(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            self.head_dim,
        )?;

        // 将注意力输出的维度交换回原来的顺序，并重塑为输出大小
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj) // 应用输出投影层
    }

    fn compute_attention_scores(
        &mut self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_dim: usize,
    ) -> Result<Tensor> {
        // 计算缩放因子（即 1/sqrt(头的维度)）
        let scale = 1f64 / f64::sqrt(head_dim as f64);

        // 计算查询与键的点积，然后乘以缩放因子
        let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        // 如果提供了注意力掩码，将其应用于注意力权重
        attn_weights = match attention_mask {
            None => attn_weights,                            // 没有掩码则直接使用权重
            Some(mask) => attn_weights.broadcast_add(mask)?, // 否则加上掩码
        };

        // 在最后一个维度上应用 softmax，计算归一化的注意力权重
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // 计算注意力加权的值张量
        attn_weights.matmul(&value_states)
    }

    // 清除键值缓存的函数
    // 改了！！！！
    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// DecoderLayer结构体的定义
#[derive(Debug, Clone)]
struct DecoderLayer {
    // 自注意力层
    self_attn: Attention,
    // 多层感知器
    mlp: MLP,
    // 输入归一化层
    input_layernorm: RmsNorm,
    // 注意力后的归一化层
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    // 创建DecoderLayer实例
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 初始化自注意力层
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        // 初始化MLP层
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        // 初始化输入层的RMS归一化
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        // 初始化注意力后的RMS归一化层
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        // 返回DecoderLayer实例
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    // 前向传播函数
    fn forward(
        &mut self,
        xs: &Tensor,                     // 输入张量
        attention_mask: Option<&Tensor>, // 可选的注意力掩码
        seqlen_offset: usize,            // 序列偏移量
        // ! 缓存总容量
        total_capacity: usize,
    ) -> Result<Tensor> {
        let residual = xs; // 保存输入张量用于残差连接
        let xs = self.input_layernorm.forward(xs)?; // 应用输入层归一化

        // ! 缓存总容量 total_capacity: usize
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, total_capacity)?; // 应用自注意力层

        let xs = (xs + residual)?; // 加上残差连接
        let residual = &xs; // 更新残差连接
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?; // 归一化并应用MLP
        residual + xs // 最终输出结果，加上残差连接
    }

    // 清除键值缓存
    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// 定义模型结构体
#[derive(Debug, Clone)]
pub struct Model {
    // 词嵌入层
    embed_tokens: candle_nn::Embedding,
    // 解码器层的列表
    layers: Vec<DecoderLayer>,
    // 最后的归一化层
    norm: RmsNorm,
    // 滑动窗口大小，用于注意力掩码
    sliding_window: usize,
    // 设备信息（如CPU或GPU）
    device: Device,
    // 数据类型
    dtype: DType,
}

impl Model {
    // 创建模型实例
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 嵌入层初始化
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        // 旋转嵌入，用于自注意力层
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        // 初始化指定数量的解码层
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        // 最后的RMS归一化层
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        // 返回模型实例
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    // 准备因果注意力掩码
    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,        // 批次大小
        tgt_len: usize,       // 目标序列长度
        seqlen_offset: usize, // 序列偏移量
    ) -> Result<Tensor> {
        // 构建因果掩码，用于防止模型关注未来的时间步
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY // 不允许的关注位置设为负无穷
                    } else {
                        0. // 允许的关注位置设为0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    // 准备一般注意力掩码
    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    // 前向传播函数
    pub fn forward(
        &mut self,
        input_ids: &Tensor,         // 输入ID张量
        seqlen_offset: usize,       // 序列偏移量
        attn_mask: Option<&Tensor>, // 可选的注意力掩码
        // ! 总缓存容量
        total_capacity: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        // 准备注意力掩码
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        // 计算嵌入
        let mut xs = self.embed_tokens.forward(input_ids)?;
        // 应用每个解码器层
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, total_capacity)?
        }
        // 最后应用归一化层
        xs.apply(&self.norm)
    }

    // 清除所有层的键值缓存
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

// 用于因果语言模型的结构体
#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model, // 基础模型
    lm_head: Linear,   // 语言模型头部，用于输出
}

impl ModelForCausalLM {
    // 创建 ModelForCausalLM 实例
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // 初始化基础模型
        let base_model = Model::new(cfg, vb.clone())?;
        // 初始化语言模型头部
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        // 返回 ModelForCausalLM 实例
        Ok(Self {
            base_model,
            lm_head,
        })
    }
    /*
    1. 函数参数解析：
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>
        input_ids: 输入的token序列，形状为 [batch_size, seq_len]
        seqlen_offset: 序列偏移量，用于位置编码
        返回值：logits张量，表示下一个token的预测概率分布

    2. 维度处理
    let (_b_size, seq_len) = input_ids.dims2()?;
    例如:
        input_ids = [[101, 2345, 3456, 4567]]
        _b_size = 1
        seq_len = 4

    */
    // 前向传播函数
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        total_capacity: usize,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        // 调用基础模型的前向传播，并应用语言模型头部
        self.base_model
            .forward(input_ids, seqlen_offset, None, total_capacity)?
            /*
            ### narrow函数的作用：
            函数用于在指定的维度上对张量进行切片，返回一个新的张量，该张量在指定维度上只包含从起始位置开始的指定长度的元素。
            ### 函数签名
            pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self>
            dim: 要进行切片的维度，类型为实现了 `Dim` trait 的类型（通常是整数）。
            start: 起始索引，从该位置开始（包含）。
            len: 切片的长度，即要取出的元素数量。
            ### 参数说明
            dim：指定对哪个维度进行操作。可以是整数或实现了 `Dim` trait 的类型。
            start：切片的起始位置，索引从 0 开始。
            len：要取出多少个元素。
            ### 使用示例
            假设有一个张量 `tensor`，其形状为 `[3, 4, 5]`：
                let tensor = ...; // 形状为 [3, 4, 5]
            如果我们想在第二个维度（即 dim = 1）上，从位置 1 开始，取出长度为 2 的切片，可以这样使用：
                let narrowed_tensor = tensor.narrow(1, 1, 2)?;
            此时，`narrowed_tensor` 的形状为 `[3, 2, 5]`。
            理解Tensor：
            !  考虑一个 3 维的张量 tensor，其形状为 [2, 4, 3]，具体数值如下，tensor: [2, 4, 3] 如下：
            let tensor = Tensor::from_vec3(&[
                ! 第一维度（dim=0）的第一个元素
                [
                    [1, 2, 3],    // tensor[0][0][*]
                    [4, 5, 6],    // tensor[0][1][*]
                    [7, 8, 9],    // tensor[0][2][*]
                    [10, 11, 12], // tensor[0][3][*]
                ],
                ! 第一维度（dim=0）的第二个元素
                [
                    [13, 14, 15], // tensor[1][0][*]
                    [16, 17, 18], // tensor[1][1][*]
                    [19, 20, 21], // tensor[1][2][*]
                    [22, 23, 24], // tensor[1][3][*]
                ],
            ]);
            ! 这个张量的形状是 [2, 4, 3]，表示：
                第一维度（dim=0，批次）：大小为 2
                第二维度（dim=1，序列长度）：大小为 4
                第三维度（dim=2，特征维度）：大小为 3
            */
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    // 清除键值缓存
    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}
