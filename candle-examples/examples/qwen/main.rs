// 删除 hf-hub 相关的依赖
// use hf_hub::{api::sync::Api, Repo, RepoType};

use anyhow::{Error as E, Result}; // 导入 `anyhow` 库的 `Error` 和 `Result`，用于错误处理
use candle::{DType, Device, Tensor}; // 导入 `candle` 库的基础类型
use candle_examples::token_output_stream::TokenOutputStream; // 导入 `TokenOutputStream`，用于文本生成
use candle_nn::VarBuilder; // 用于构建和加载神经网络的权重
use candle_transformers::generation::LogitsProcessor; // 用于处理 logits，调整生成概率
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase}; // 导入 Qwen 基础模型
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe}; // 导入 Qwen MoE 模型
use clap::Parser; // 使用 `clap` 库解析命令行参数
use tokenizers::Tokenizer; // 分词器库，用于文本编码和解码

// 定义模型类型的枚举：支持基础模型和 MoE 模型
enum Model {
    Base(ModelBase), // 基础模型
    Moe(ModelMoe),   // MoE 模型
}

// 为 `Model` 实现 `forward` 方法，根据模型类型执行前向传播
impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize, total_capacity: usize) -> candle::Result<Tensor> {
        match self {
            Self::Moe(ref mut m) => m.forward(xs, s), // 使用 MoE 模型的 `forward`
            // 用的是普通模型
            Self::Base(ref mut m) => m.forward(xs, s, total_capacity), // 使用基础模型的 `forward`
        }
    }
}

// 文本生成器结构体，包含模型、设备、分词器等组件
struct TextGeneration {
    model: Model,                      // 文本生成模型
    device: Device,                    // 运行设备（CPU 或 GPU）
    tokenizer: TokenOutputStream,      // 分词器输出流，用于文本生成
    logits_processor: LogitsProcessor, // logits 处理器，用于调整生成概率
    repeat_penalty: f32,               // 重复惩罚系数
    repeat_last_n: usize,              // 重复检测的最大 token 数
}

// 为 `TextGeneration` 实现初始化和运行文本生成的方法
impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p); // 初始化 logits 处理器
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer), // 初始化分词器输出流
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    // 执行文本生成逻辑！！！！！！！！！！
    // 以下也可能用到cache
    // ！！！！！！！！！！！！！！！！！！！
    fn run(&mut self, prompt: &str, sample_len: usize, total_capacity: usize) -> Result<()> {
        use std::io::Write; // 引入 `Write` trait 用于控制台输出

        // 1. 清空分词器的历史状态，防止前序调用中的遗留内容影响当前生成
        self.tokenizer.clear();

        // 2. 编码提示文本，将字符串 `prompt` 转换为 token ID 列表
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true) // 对输入文本进行编码，`true`表示保留特殊tokens
            .map_err(E::msg)?
            .get_ids() // 提取编码结果中的 token ID
            .to_vec(); // 转换为可变的 `Vec` 以便后续处理

        // 3. 输出初始 token（提示文本的 token）以便用户查看生成的初始状态
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}");
            }
        }
        std::io::stdout().flush()?; // 刷新输出缓冲区

        // 4. 初始化生成统计
        let mut generated_tokens = 0usize; // 记录已生成 token 数量
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,                                         // 定义终止 token
            None => anyhow::bail!("cannot find the <|endoftext|> token"), // 若未找到则报错
        };
        let start_gen = std::time::Instant::now(); // 记录生成起始时间

        //let mut memory_hog = Vec::new(); // ! 用来模拟内存占用的 vector

        // 5. 开始生成文本
        for index in 0..sample_len {
            //memory_hog.push(vec![0u8; 10_000_000]); // ! 每次分配大约 10MB 的内存
            //print!("\n增加10mb！！！"); // !

            // 设定上下文长度：生成第一个 token 时包含所有提示，之后仅包含最后一个 token
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size); // 计算上下文的起始位置
            let ctxt = &tokens[start_pos..]; // 获取当前上下文的 tokens

            // 6. 将上下文转换为张量，并添加 batch 维度（通常为1）
            /*
            这个函数是 unsqueeze 操作的实现，主要用于在张量的指定维度插入一个大小为1的新维度。

            示例：
            假设原始张量形状是 [3, 4]
            tensor.unsqueeze(1)?;  // 结果形状变为 [3, 1, 4]

            关键步骤：
            1.获取原始维度和步长
            2.插入新维度
            3.处理步长
            4.创建新张量：
                保持原始数据存储不变
                只修改形状和步长信息
                设置反向传播操作为 Reshape

            */
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // 7. 调用模型的 `forward` 方法进行前向传播，获取 logits
            // 以下也可能用到cache
            // ！！！！！！！！！！！！！！！！！！！
            let logits = self.model.forward(&input, start_pos, total_capacity)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?; // 调整 logits 维度

            // 8. 应用重复惩罚以减少重复生成（若 `repeat_penalty` 不为 1）
            // 以下也可能用到cache
            // ！！！！！！！！！！！！！！！！！！！
            let logits = if self.repeat_penalty == 1. {
                logits // 若惩罚为1则不修改 logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n); // 重复惩罚窗口起点
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..], // 应用于窗口内的 token
                )?
            };

            // 9. 通过采样机制生成下一个 token
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token); // 将生成的 token 添加到 tokens 列表
            generated_tokens += 1; // 更新已生成 token 数量

            // 10. 若生成了终止 token 则退出循环
            // !!!!!!!!!!!!!略作修改
            // ! 可以删除，使得无限输出tokens
            if next_token == eos_token {
                break;
            }

            // 11. 输出生成的 token
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            // !!!!!!!!!!!!!!!!!

            // 每生成100个词记录一次
            if generated_tokens % 100 == 0 {
                let dt = start_gen.elapsed(); // 计算生成总时间
                print!(
                    "\x1b[31m$@#现在已经运行了{:?}，已经生成了{}个tokens，平均生成速度是{:.2} token/s!$@#\x1b[0m",
                    dt,
                    generated_tokens,
                    generated_tokens as f64 / dt.as_secs_f64(), // 计算生成速度
                );
            }
            // !!!!!!!!!!!!!!!!!
        }

        // 12. 输出生成所花时间及速度
        let dt = start_gen.elapsed(); // 计算生成总时间
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}"); // 输出未解码的 tokens
        }
        std::io::stdout().flush()?;
        print!(
            "\x1b[31m$@#现在已经运行了{:?}，已经生成了{}个tokens，平均生成速度是{:.2} token/s!$@#\x1b[0m",
            dt,
            generated_tokens,
            generated_tokens as f64 / dt.as_secs_f64(), // 计算生成速度
        );
        Ok(())
    }
}

// 定义支持的模型类型
#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)] // 自动实现拷贝、调试打印、比较等特性
enum WhichModel {
    #[value(name = "0.5b")]
    W0_5b, // 模型参数大小为 0.5 billion
    #[value(name = "1.8b")]
    W1_8b, // 模型参数大小为 1.8 billion
    #[value(name = "4b")]
    W4b, // 模型参数大小为 4 billion
    #[value(name = "7b")]
    W7b, // 模型参数大小为 7 billion
    #[value(name = "14b")]
    W14b, // 模型参数大小为 14 billion
    #[value(name = "72b")]
    W72b, // 模型参数大小为 72 billion
    #[value(name = "moe-a2.7b")]
    MoeA27b, // 使用混合专家模型（MoE）架构，参数大小为 2.7 billion
    #[value(name = "2-0.5b")]
    W2_0_5b, // Qwen 第二代模型，参数大小为 0.5 billion
    #[value(name = "2-1.5b")]
    W2_1_5b, // Qwen 第二代模型，参数大小为 1.5 billion
    #[value(name = "2-7b")]
    W2_7b, // Qwen 第二代模型，参数大小为 7 billion
    #[value(name = "2-72b")]
    W2_72b, // Qwen 第二代模型，参数大小为 72 billion
}

// 解析命令行参数的结构体，包含文本生成的所有配置项
#[derive(Parser, Debug)] // 使用 `clap` 库的 `Parser` 宏生成命令行解析器
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)] // `--cpu` 选项，指定是否使用 CPU
    cpu: bool,

    #[arg(long)] // `--tracing` 选项，启用追踪日志
    tracing: bool,

    #[arg(long)] // `--use-flash-attn` 选项，启用闪存注意力机制
    use_flash_attn: bool,

    #[arg(long)] // `--prompt` 选项，必选参数，生成的起始提示文本
    prompt: String,

    #[arg(long)] // `--temperature` 选项，指定生成温度（可选）
    temperature: Option<f64>,

    #[arg(long)] // `--top-p` 选项，指定 top-p 采样阈值（可选）
    top_p: Option<f64>,

    #[arg(long, default_value_t = 299792458)] // `--seed` 选项，设置随机数种子，默认值为 299792458
    seed: u64,

    #[arg(long, short = 'n', default_value_t = 10000)]
    // `--sample-len` 选项，生成文本的最大长度，默认为 10000
    sample_len: usize,

    #[arg(long)] // `--model-id` 选项，指定 Hugging Face 模型仓库的 ID
    model_id: Option<String>,

    #[arg(long, default_value = "main")] // `--revision` 选项，指定模型的版本修订号，默认为 "main"
    revision: String,

    #[arg(long)] // `--tokenizer-file` 选项，指定分词器文件的路径
    tokenizer_file: Option<String>,

    #[arg(long)] // `--weight-files` 选项，指定权重文件路径
    weight_files: Option<String>,

    #[arg(long, default_value_t = 1.5)] // `--repeat-penalty` 选项，指定重复惩罚系数，默认值为 1.1
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    // `--repeat-last-n` 选项，用于重复惩罚的最后 token 数量，默认值为 64
    repeat_last_n: usize,

    #[arg(long, default_value_t = 35)]
    // total_capacity 缓存算法总容量
    total_capacity: usize,

    #[arg(long, default_value = "0.5b")] // `--model` 选项，指定要使用的模型类型，默认为 "0.5b"
    model: WhichModel,
}

// 主函数入口
fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder; // 引入 `ChromeLayerBuilder`，用于性能追踪
    use tracing_subscriber::prelude::*; // 引入 `tracing_subscriber` 的相关模块，用于日志追踪

    let args = Args::parse(); // 解析命令行参数，生成 `Args` 结构体实例
    let _guard = if args.tracing {
        // 如果启用了追踪（`--tracing`），初始化 ChromeLayer 以记录追踪数据
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init(); // 将 chrome_layer 注册到日志订阅者中
        Some(guard) // 返回 guard 以保持追踪状态
    } else {
        None // 未启用追踪时，不进行任何操作
    };

    // 检查可用的硬件特性并打印
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),     // 检查是否支持 AVX 指令集
        candle::utils::with_neon(),    // 检查是否支持 NEON 指令集
        candle::utils::with_simd128(), // 检查是否支持 SIMD128 指令集
        candle::utils::with_f16c()     // 检查是否支持 f16c 指令集
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.), // 打印温度参数（若未设置则默认值为 0）
        args.repeat_penalty,            // 打印重复惩罚参数
        args.repeat_last_n              // 打印重复检测的 token 数目
    );

    let start = std::time::Instant::now(); // 记录时间，计算加载过程耗时

    // 模型和分词器的本地文件路径
    let model_dir = "/Qwen_Qwen1.5-0.5B";
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file), // 使用用户指定的分词器文件
        None => std::path::PathBuf::from(format!("{}/tokenizer.json", model_dir)), // 若未指定，则使用默认路径
    };

    // 确定模型权重文件路径
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',') // 将用户指定的文件路径按逗号分隔
            .map(std::path::PathBuf::from) // 将每个路径转换为 `PathBuf`
            .collect::<Vec<_>>(), // 转换为 `Vec<PathBuf>` 类型
        None => vec![std::path::PathBuf::from(format!(
            "{}/model.safetensors",
            model_dir // 若未指定权重路径，则使用默认路径
        ))],
    };

    println!("retrieved the files in {:?}", start.elapsed()); // 打印检索文件的耗时
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?; // 加载分词器，若失败则返回错误

    let start = std::time::Instant::now(); // 记录加载配置和模型开始时间
    let config_file = std::path::PathBuf::from(format!("{}/config.json", model_dir)); // 模型配置文件的路径
    let device = candle_examples::device(args.cpu)?; // 选择运行设备，取决于 `--cpu` 参数
    let dtype = if device.is_cuda() {
        DType::BF16 // 如果设备是 CUDA，则使用 BF16 数据类型
    } else {
        DType::F32 // 否则使用 F32 数据类型
    };

    // 加载模型权重文件
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    // 根据命令行参数 `model` 字段选择模型类型并加载配置
    let model = match args.model {
        WhichModel::MoeA27b => {
            // 如果选择了 MoE 模型，加载 MoE 配置
            let config: ConfigMoe = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Moe(ModelMoe::new(&config, vb)?) // 初始化 MoE 模型
        }
        _ => {
            // 如果选择了基础模型，加载基础模型的配置
            let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Base(ModelBase::new(&config, vb)?) // 初始化基础模型
        }
    };

    println!("loaded the model in {:?}", start.elapsed()); // 打印加载模型的耗时

    // 初始化文本生成管道
    let mut pipeline = TextGeneration::new(
        model,               // 设置已加载的模型
        tokenizer,           // 设置分词器
        args.seed,           // 设置随机种子
        args.temperature,    // 设置温度参数
        args.top_p,          // 设置 top-p 采样阈值
        args.repeat_penalty, // 设置重复惩罚系数
        args.repeat_last_n,  // 设置用于重复检测的 token 数目
        &device,             // 指定设备
    );
    pipeline.run(&args.prompt, args.sample_len, args.total_capacity)?; // 使用生成管道生成文本并输出
    Ok(())
}
