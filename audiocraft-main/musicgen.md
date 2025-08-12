# MusicGen 模型详细文档

## 1. 模型定义

### 1.1 概述

MusicGen 是一个简单且可控的音乐生成模型，由 AudioCraft 提供代码和模型支持。它是一个单阶段自回归 Transformer 模型，基于 32kHz 的 EnCodec 编码器进行训练，使用 4 个代码簿以 50 Hz 的频率采样。

**相关代码文件**：

- `audiocraft/models/musicgen.py` - MusicGen 模型的核心实现。

### 1.2 特点

- **无需自监督语义表示**：不像现有的方法如 MusicLM，MusicGen 不需要自监督语义表示。
- **并行生成**：通过在代码簿之间引入小延迟，MusicGen 可以并行预测所有 4 个代码簿，因此每秒音频只需要 50 个自回归步骤。
- **数据集**：使用 20K 小时的许可音乐进行训练，包括内部数据集的 10K 高质量音乐曲目，以及 ShutterStock 和 Pond5 音乐数据。

**相关代码文件**：

- `audiocraft/data/audio.py` - 音频数据处理和数据集准备。

### 1.3 提供的模型

AudioCraft 提供了以下预训练模型：

- `facebook/musicgen-small`：300M 参数，仅支持文本到音乐。
- `facebook/musicgen-medium`：1.5B 参数，仅支持文本到音乐。
- `facebook/musicgen-melody`：1.5B 参数，支持文本到音乐和文本+旋律到音乐。
- `facebook/musicgen-large`：3.3B 参数，仅支持文本到音乐。
- `facebook/musicgen-melody-large`：3.3B 参数，支持文本到音乐和文本+旋律到音乐。
- `facebook/musicgen-stereo-*`：所有上述模型的立体声生成微调版本。

最佳质量与计算平衡的模型是 `facebook/musicgen-medium` 或 `facebook/musicgen-melody`。

**相关代码文件**：

- `audiocraft/models/musicgen.py` - 包含所有预训练模型的加载和初始化逻辑。

### 1.4 硬件要求

运行 MusicGen 模型需要至少 16GB 内存的 GPU。较小的 GPU 可以使用 `facebook/musicgen-small` 模型生成短序列。

**相关代码文件**：

- 无特定代码文件，但硬件要求在模型初始化和生成过程中会影响性能。

## 2. 模型的训练

### 2.1 训练管道

MusicGen 的训练管道由 `MusicGenSolver` 实现，定义了一个自回归语言建模任务，处理从预训练 EnCodec 模型提取的多个离散令牌流。

**相关代码文件**：

- `audiocraft/solvers/musicgen.py` - MusicGenSolver 的实现。

### 2.2 数据集准备

- **数据集**：AudioCraft 不提供用于训练 MusicGen 的数据集，需要用户准备自己的数据集。
- **元数据**：数据集需要包含音乐特定的元数据，以 `.json` 文件格式存储在音频文件相同的位置。

**相关代码文件**：

- `audiocraft/data/audio_dataset.py` - 数据集和元数据的处理逻辑。

### 2.3 训练配置和网格

- **配置文件**：MusicGen 提供了多种配置文件，例如：
  - 文本到音乐的基础模型：`solver=musicgen/musicgen_base_32khz`
  - 支持色谱图条件的模型：`solver=musicgen/musicgen_melody_32khz`
- **模型规模**：提供三种规模：`small` (300M)、`medium` (1.5B)、`large` (3.3B)。
- **示例网格**：可以在 `audiocraft/grids/musicgen/` 找到，用于训练 MusicGen 模型。
  ```bash
  # 文本到音乐
  dora grid musicgen.musicgen_base_32khz --dry_run --init
  # 旋律引导的音乐生成
  dora grid musicgen.musicgen_melody_base_32khz --dry_run --init
  ```

**相关代码文件**：

- `config/solver/musicgen/` - 配置文件目录。
- `audiocraft/grids/musicgen/` - 示例网格目录。

### 2.4 音频令牌化器

支持多种音频令牌化器，如预训练的 EnCodec 模型、DAC 或用户自己的模型。通过 `compression_model_checkpoint` 设置控制。

**相关代码文件**：

- `audiocraft/models/encodec.py` - EnCodec 模型的实现。

### 2.5 立体声模型训练

通过设置 `interleave_stereo_codebooks.use=True` 和 `channels=2` 激活立体声训练。

**相关代码文件**：

- `audiocraft/solvers/musicgen.py` - 包含立体声训练的配置和实现。

### 2.6 微调现有模型

可以通过 `continue_from` 参数初始化模型到预训练模型或之前训练的模型。

**相关代码文件**：

- `audiocraft/train.py` - 包含模型微调的逻辑。

### 2.7 缓存 EnCodec 令牌

可以预计算 EnCodec 令牌和其他元数据以加速训练。

**相关代码文件**：

- `audiocraft/data/audio_dataset.py` - 包含缓存令牌的逻辑。

### 2.8 评估阶段

评估阶段默认计算交叉熵和困惑度，可以通过配置文件启用目标度量。

**相关代码文件**：

- `audiocraft/solvers/musicgen.py` - 包含评估逻辑。
- `config/solver/musicgen/evaluation/` - 评估配置文件目录。

### 2.9 生成阶段

生成阶段允许有条件或无条件地生成样本，支持多种采样方法（贪婪、softmax 采样、top-K、top-P）。

**相关代码文件**：

- `audiocraft/models/musicgen.py` - 包含生成逻辑。

## 3. 模型的测试

### 3.1 使用方式

AudioCraft 提供了多种与 MusicGen 交互的方式：

1. **Hugging Face 空间**：在 [`facebook/MusicGen` Hugging Face Space](https://huggingface.co/spaces/facebook/MusicGen) 上有演示。
2. **Colab 笔记本**：可以在 Colab 上运行扩展演示。
3. **本地 Gradio 演示**：运行 `python -m demos.musicgen_app --share`。
4. **Jupyter 笔记本**：在本地运行 `demos/musicgen_demo.ipynb`（需要 GPU）。
5. **社区 Colab 页面**：查看 [@camenduru Colab page](https://github.com/camenduru/MusicGen-colab)。

**相关代码文件**：

- `demos/musicgen_app.py` - 本地 Gradio 演示应用。
- `demos/musicgen_demo.ipynb` - Jupyter 笔记本演示。

### 3.2 API 使用示例

以下是一个简单的 API 使用示例，用于生成音乐：

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # 生成 8 秒的音频
wav = model.generate_unconditional(4)    # 生成 4 个无条件音频样本
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # 生成 3 个样本

melody, sr = torchaudio.load('./assets/bach.mp3')
# 使用给定的音频旋律和描述生成音乐
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # 将保存为 {idx}.wav，音量标准化为 -14 db LUFS
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

**相关代码文件**：

- `audiocraft/models/musicgen.py` - MusicGen 模型的 API 实现。
- `audiocraft/data/audio.py` - 音频处理和保存逻辑。

### 3.3 使用 Hugging Face Transformers 库

MusicGen 也可以通过 Transformers 库使用：

1. 安装 Transformers 库：
   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   ```
2. 生成音频样本：
   ```python
   from transformers import AutoProcessor, MusicgenForConditionalGeneration

   processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
   model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

   inputs = processor(
       text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
       padding=True,
       return_tensors="pt",
   )

   audio_values = model.generate(**inputs, max_new_tokens=256)
   ```
3. 收听或保存音频样本：
   - 在 IPython 笔记本中收听：
     ```python
     from IPython.display import Audio

     sampling_rate = model.config.audio_encoder.sampling_rate
     Audio(audio_values[0].numpy(), rate=sampling_rate)
     ```
   - 保存为 `.wav` 文件：
     ```python
     import scipy

     sampling_rate = model.config.audio_encoder.sampling_rate
     scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
     ```

**相关代码文件**：

- 无特定代码文件，但可以使用 Transformers 库的 API。

### 3.4 收听样本

可以通过 MOS 工具收听和比较模型生成的样本：

```bash
gunicorn -w 4 -b 127.0.0.1:8895 -t 120 'scripts.mos:app' --access-logfile -
```

然后访问 [https://127.0.0.1:8895](https://127.0.0.1:8895)。

**相关代码文件**：

- `scripts/mos.py` - MOS 工具的实现。

### 3.5 导入/导出模型

可以通过特定脚本导出模型以兼容 `audiocraft.models.MusicGen` API。

**相关代码文件**：

- `audiocraft/utils/export.py` - 模型导入和导出的工具函数。

## 4. 常见问题解答 (FAQ)

- **训练代码是否可用？** 是的，AudioCraft 提供了 EnCodec、MusicGen、Multi Band Diffusion 和 JASCO 的训练代码。
- **模型存储在哪里？** 模型存储在 Hugging Face 指定的位置，可以通过环境变量 `AUDIOCRAFT_CACHE_DIR` 覆盖。
- **Windows 用户帮助**：可以参考 @FurkanGozukara 的教程 [AudioCraft/MusicGen on Windows](https://youtu.be/v-YpvPkhdO4)。
- **Colab 演示帮助**：可以查看 @camenduru 的教程 [YouTube](https://www.youtube.com/watch?v=EGfxuTy9Eeo)。
- **参数解释**：有关 `top-k`、`top-p`、`temperature` 和 `classifier-free guidance` 的解释，请参考 @FurkanGozukara 的教程 [GitHub](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Tutorials/AI-Music-Generation-Audiocraft-Tutorial.md#more-info-about-top-k-top-p-temperature-and-classifier-free-guidance-from-chatgpt)。

## 5. 引用

如果您使用 MusicGen 模型，请引用以下论文：

```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

## 6. 许可证

- 代码许可证：MIT 许可证。
- 模型权重许可证：CC-BY-NC 4.0 许可证。



您好！关于 `musicgen_app.py` 文件，这是一个用于运行 MusicGen 模型的 Gradio 界面应用程序。以下是对其测试方法、运行时逻辑以及数据流的详细解释。

### 如何进行测试

要测试 `musicgen_app.py`，您需要按照以下步骤操作：

1. **确保环境配置正确**：

   - 确保您已经安装了必要的依赖项，包括 `audiocraft`、`torch`、`gradio` 等。
   - 确保您的系统有足够的 GPU 内存（至少 16GB）来运行 MusicGen 模型。
2. **运行应用程序**：

   - 在终端中导航到包含 `musicgen_app.py` 的目录（例如 `E:\python_project\Alex项目\audiocraft-main\audiocraft-main\demos`）。
   - 运行以下命令启动应用程序：
     ```bash
     python -m demos.musicgen_app --share
     ```
   - `--share` 参数会生成一个公共链接，允许您通过浏览器访问界面。如果不使用此参数，界面将在本地运行，默认地址为 `http://127.0.0.1:7860`。
3. **访问界面**：

   - 运行命令后，终端会显示一个 URL（例如 `http://127.0.0.1:7860` 或一个公共链接）。
   - 在浏览器中打开此 URL，您将看到 MusicGen 的 Gradio 界面。
4. **测试功能**：

   - **输入文本**：在文本框中输入描述音乐的文本（例如“An 80s driving pop song with heavy drums and synth pads in the background”）。
   - **条件旋律（可选）**：选择“file”或“mic”来上传音频文件或使用麦克风录制旋律。
   - **选择模型**：从下拉菜单中选择一个预训练模型（例如 `facebook/musicgen-stereo-melody`）。
   - **设置参数**：调整生成参数，如时长（`duration`）、Top-k、Top-p、温度（`temperature`）和分类器自由引导（`cfg_coef`）。
   - **提交生成**：点击“Submit”按钮生成音乐。
   - **查看结果**：生成完成后，界面会显示一个视频和音频文件，您可以播放或下载。
5. **中断生成**：

   - 如果生成过程耗时过长，您可以点击“Interrupt”按钮中断过程。

### 代码运行时的逻辑

`musicgen_app.py` 的运行逻辑如下：

1. **初始化**：

   - 代码首先定义了一些全局变量，如 `MODEL`（用于存储当前加载的模型）、`IS_BATCHED`（是否为批量模式）等。
   - 设置了一个进程池（`ProcessPoolExecutor`）用于并行处理视频生成。
2. **模型加载**：

   - `load_model` 函数负责加载指定的 MusicGen 模型。如果当前模型与请求的模型不同，它会清除旧模型并加载新模型。
   - `load_diffusion` 函数加载 MultiBand Diffusion 解码器（如果选择使用）。
3. **生成音乐**：

   - `predict_full` 函数是主要的生成逻辑：
     - 首先检查输入参数（如温度、Top-k、Top-p 是否有效）。
     - 加载指定的模型和解码器。
     - 设置生成参数（如时长、Top-k、Top-p、温度等）。
     - 调用 `_do_predictions` 函数生成音乐。
   - `_do_predictions` 函数处理文本和旋律输入，调用模型的生成方法（`generate` 或 `generate_with_chroma`），并处理输出（包括使用 MultiBand Diffusion 解码器如果启用）。
   - 生成的音频被保存为临时 WAV 文件，并转换为视频文件（使用 `make_waveform` 函数）。
4. **界面定义**：

   - `ui_full` 函数定义了完整的 Gradio 界面，包括输入字段、按钮、输出显示区域和示例。
   - `ui_batched` 函数定义了一个简化的批量处理界面，用于处理多个输入。
5. **事件处理**：

   - 界面中的按钮（如“Submit”）和选择框（如“file”或“mic”）会触发相应的函数（如 `predict_full`、`toggle_audio_src`）。
6. **启动应用程序**：

   - 代码的 `if __name__ == "__main__":` 部分解析命令行参数，并根据是否为批量模式启动相应的界面（`ui_full` 或 `ui_batched`）。

### 数据流如何流向

数据流在 `musicgen_app.py` 中的流向如下：

1. **用户输入**：

   - 用户通过 Gradio 界面输入文本描述、上传旋律音频（可选）、选择模型和设置生成参数。
2. **输入处理**：

   - 文本输入被截断到最大长度（512 个字符）。
   - 旋律音频（如果提供）被转换为目标采样率（32kHz）和单声道。
3. **模型生成**：

   - 输入数据（文本和旋律）被传递给 `predict_full` 函数。
   - 根据是否提供旋律，调用 `MODEL.generate`（仅文本）或 `MODEL.generate_with_chroma`（文本+旋律）生成音频令牌。
   - 如果启用 MultiBand Diffusion，生成的令牌被进一步处理为音频波形。
4. **输出处理**：

   - 生成的音频被保存为临时 WAV 文件。
   - WAV 文件被转换为视频文件（包含波形可视化），使用并行处理的进程池。
5. **结果显示**：

   - 生成的视频和音频文件路径被返回到 Gradio 界面，并在输出区域显示。
   - 临时文件被添加到文件清理器（`FileCleaner`）中，以便在一定时间后自动删除。
6. **用户交互**：

   - 用户可以播放生成的音乐视频或下载音频文件。
   - 用户可以中断生成过程或重新提交新的输入。

希望以上解释能帮助您理解 `musicgen_app.py` 的测试方法、运行逻辑和数据流。如果您有任何其他问题或需要进一步的帮助，请随时告诉我！
