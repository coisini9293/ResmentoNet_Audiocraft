# 版权声明：本代码由 Meta Platforms, Inc. 及其附属公司拥有版权。
# 版权所有。
# 许可证声明：本源代码根据源代码根目录中的 LICENSE 文件中的许可证进行许可。
# 更新说明：根据 https://github.com/rkfg/audiocraft/blob/long/app.py 进行的 UI 更改更新
# 同样根据 MIT 许可证发布。

import argparse  # 导入 argparse 模块，用于解析命令行参数
import logging  # 导入 logging 模块，用于日志记录
import os  # 导入 os 模块，用于操作系统接口功能，如环境变量获取
import subprocess as sp  # 导入 subprocess 模块，用于运行外部命令
import sys  # 导入 sys 模块，用于系统特定的参数和功能
import time  # 导入 time 模块，用于时间相关功能
import typing as tp  # 导入 typing 模块，用于类型注解
import warnings  # 导入 warnings 模块，用于警告控制
from pathlib import Path  # 导入 Path，用于文件路径处理
from tempfile import NamedTemporaryFile  # 导入 NamedTemporaryFile，用于创建临时文件
import uuid  # 生成唯一文件名
import shutil  # 文件复制/移动

from concurrent.futures import ProcessPoolExecutor  # 从 concurrent.futures 导入 ProcessPoolExecutor，用于并行处理
from einops import rearrange  # 从 einops 导入 rearrange 函数，用于张量重排
import torch  # 导入 torch 模块，用于 PyTorch 深度学习框架
import gradio as gr  # 导入 gradio 模块，用于创建交互式界面
# 兼容处理：尝试导入新版 gradio_client 的波形工具，不存在则走 ffmpeg 回退
try:
    from gradio_client import media_utils as _gradio_media_utils  # type: ignore
except Exception:  # pragma: no cover - 某些版本不存在该模块
    _gradio_media_utils = None

from audiocraft.data.audio_utils import convert_audio  # 从 audiocraft.data.audio_utils 导入 convert_audio 函数，用于音频转换
from audiocraft.data.audio import audio_write  # 从 audiocraft.data.audio 导入 audio_write 函数，用于音频文件写入
from audiocraft.models.encodec import InterleaveStereoCompressionModel  # 从 audiocraft.models.encodec 导入 InterleaveStereoCompressionModel 类，用于立体声压缩模型
from audiocraft.models import MusicGen, MultiBandDiffusion  # 从 audiocraft.models 导入 MusicGen 和 MultiBandDiffusion 类，用于音乐生成和多频带扩散模型


MODEL = None  # 定义全局变量 MODEL，用于存储最后使用的模型
SPACE_ID = os.environ.get('SPACE_ID', '')  # 获取环境变量 SPACE_ID，如果不存在则为空字符串
IS_BATCHED = "facebook/MusicGen" in SPACE_ID or 'musicgen-internal/musicgen_dev' in SPACE_ID  # 判断是否为批量模式，基于 SPACE_ID 是否包含特定字符串
print(IS_BATCHED)  # 打印是否为批量模式的标志
MAX_BATCH_SIZE = 12  # 定义最大批量大小为 12
BATCHED_DURATION = 15  # 定义批量模式的生成时长为 15 秒
INTERRUPTING = False  # 定义全局变量 INTERRUPTING，用于标记是否中断生成过程
MBD = None  # 定义全局变量 MBD，用于存储 MultiBandDiffusion 模型
# 我们必须包装 subprocess 调用，以在使用 gr.make_waveform 时清理日志
_old_call = sp.call  # 保存 subprocess.call 的原始函数引用


def _call_nostderr(*args, **kwargs):  # 定义 _call_nostderr 函数，包装 subprocess.call 以隐藏 ffmpeg 的日志输出
    # 避免 ffmpeg 在日志中输出过多信息
    kwargs['stderr'] = sp.DEVNULL  # 将 stderr 设置为 DEVNULL，隐藏错误输出
    kwargs['stdout'] = sp.DEVNULL  # 将 stdout 设置为 DEVNULL，隐藏标准输出
    _old_call(*args, **kwargs)  # 调用原始的 subprocess.call 函数


sp.call = _call_nostderr  # 将 subprocess.call 替换为自定义的 _call_nostderr 函数
# 预分配进程池，用于并行处理
pool = ProcessPoolExecutor(4)  # 创建一个包含 4 个进程的进程池
pool.__enter__()  # 进入进程池的上下文管理器，启动进程池

# 结果输出目录（默认当前工作目录下的 result，可用环境变量 RESULT_DIR 覆盖）
RESULT_DIR = Path(os.environ.get('RESULT_DIR', 'result')).resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def interrupt():  # 定义 interrupt 函数，用于设置中断标志
    global INTERRUPTING  # 声明使用全局变量 INTERRUPTING
    INTERRUPTING = True  # 设置中断标志为 True


class FileCleaner:  # 定义 FileCleaner 类，用于管理临时文件并在一定时间后清理
    def __init__(self, file_lifetime: float = 3600):  # 初始化方法，设置文件生存时间默认为 3600 秒（1小时）
        self.file_lifetime = file_lifetime  # 保存文件生存时间
        self.files = []  # 初始化文件列表，用于存储文件路径和添加时间

    def add(self, path: tp.Union[str, Path]):  # 定义 add 方法，用于添加文件到清理列表
        self._cleanup()  # 调用清理方法，删除过期的文件
        self.files.append((time.time(), Path(path)))  # 将当前时间和文件路径添加到文件列表

    def _cleanup(self):  # 定义 _cleanup 方法，用于清理过期的文件
        now = time.time()  # 获取当前时间
        for time_added, path in list(self.files):  # 遍历文件列表
            if now - time_added > self.file_lifetime:  # 如果文件添加时间距现在超过生存时间
                if path.exists():  # 如果文件仍然存在
                    path.unlink()  # 删除文件
                self.files.pop(0)  # 从列表中移除该文件记录
            else:  # 如果文件未过期
                break  # 停止遍历，因为后续文件也不会过期
                
file_cleaner = FileCleaner()  # 创建 FileCleaner 实例，用于管理临时文件


def _make_waveform_ffmpeg(input_audio_path: str) -> str:
    """使用 ffmpeg 生成带音频的动态波形视频，返回生成的 .mp4 路径。"""
    if not input_audio_path:
        raise ValueError("input_audio_path is required for waveform rendering")
    # 输出到 RESULT_DIR，文件名与音频同名
    in_stem = Path(input_audio_path).stem or str(uuid.uuid4())
    output_video_path = str((RESULT_DIR / f"{in_stem}.mp4").resolve())
    # 动态波形，浅色线条；确保 yuv420p 以兼容大多数播放器
    ff_filter = "[0:a]showwaves=s=1280x360:mode=cline:colors=white,format=yuv420p[v]"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_audio_path,
        "-filter_complex", ff_filter,
        "-map", "[v]", "-map", "0:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest",
        output_video_path,
    ]
    try:
        sp.call(cmd)
    except Exception as exc:
        # 回落到仅返回音频路径（依旧可在 UI 播放）
        return input_audio_path
    return output_video_path


def make_waveform(*args, **kwargs):  # 定义 make_waveform 函数，用于创建音频波形视频
    # 进一步移除一些警告信息
    be = time.time()  # 记录开始时间
    with warnings.catch_warnings():  # 使用警告捕获上下文管理器
        warnings.simplefilter('ignore')  # 忽略警告
        if _gradio_media_utils is not None:
            out = _gradio_media_utils.make_waveform(*args, **kwargs)  # 优先使用新版 gradio 的工具
            # 将生成的视频复制/移动到 RESULT_DIR，命名与音频一致
            try:
                in_path = Path(args[0] if args else kwargs.get('input', ''))
                target = RESULT_DIR / f"{in_path.stem}.mp4"
                shutil.copy2(out, target)
                out = str(target.resolve())
            except Exception:
                pass
        else:
            # 兼容旧调用：只传入音频路径
            input_audio_path = args[0] if args else kwargs.get("input", "")
            out = _make_waveform_ffmpeg(str(input_audio_path))
        print("Make a video took", time.time() - be)  # 打印创建视频所花费的时间
        return out  # 返回生成的视频路径


def load_model(version='facebook/musicgen-melody'):  # 定义 load_model 函数，用于加载指定版本的 MusicGen 模型
    global MODEL  # 声明使用全局变量 MODEL
    print("Loading model", version)  # 打印正在加载的模型版本
    
    if MODEL is None or MODEL.name != version:  # 如果 MODEL 未定义或当前模型版本与请求版本不同
        # 清除 PyTorch CUDA 缓存并删除模型
        del MODEL  # 删除当前模型
        torch.cuda.empty_cache()  # 清除 CUDA 缓存
        MODEL = None  # 将 MODEL 设置为 None，以防加载过程中崩溃
        MODEL = MusicGen.get_pretrained(version)  # 加载指定版本的预训练模型


def load_diffusion():  # 定义 load_diffusion 函数，用于加载 MultiBandDiffusion 模型
    global MBD  # 声明使用全局变量 MBD
    if MBD is None:  # 如果 MBD 未定义
        print("loading MBD")  # 打印正在加载 MBD 的消息
        MBD = MultiBandDiffusion.get_mbd_musicgen()  # 加载 MultiBandDiffusion 模型


def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None, **gen_kwargs):  # 定义 _do_predictions 函数，用于执行音乐生成预测
    MODEL.set_generation_params(duration=duration, **gen_kwargs)  # 设置模型的生成参数，包括时长和其他参数
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])  # 打印新批次的信息，包括文本数量、文本内容和旋律信息
    be = time.time()  # 记录开始时间
    processed_melodies = []  # 初始化处理后的旋律列表
    target_sr = 32000  # 定义目标采样率为 32000 Hz
    target_ac = 1  # 定义目标音频通道数为 1（单声道）
    for melody in melodies:  # 遍历输入的旋律列表
        if melody is None:  # 如果旋律为 None
            processed_melodies.append(None)  # 添加 None 到处理后的旋律列表
        else:  # 如果旋律存在
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()  # 获取采样率和旋律数据，并转换为 PyTorch 张量
            if melody.dim() == 1:  # 如果旋律是一维张量
                melody = melody[None]  # 增加一个维度，变为二维张量
            melody = melody[..., :int(sr * duration)]  # 截取旋律数据，确保长度不超过指定时长
            melody = convert_audio(melody, sr, target_sr, target_ac)  # 转换旋律音频到目标采样率和通道数
            processed_melodies.append(melody)  # 添加处理后的旋律到列表

    try:  # 尝试执行生成过程
        if any(m is not None for m in processed_melodies):  # 如果存在非空的旋律
            outputs = MODEL.generate_with_chroma(  # 使用旋律条件生成音乐
                descriptions=texts,  # 文本描述
                melody_wavs=processed_melodies,  # 处理后的旋律波形
                melody_sample_rate=target_sr,  # 旋律采样率
                progress=progress,  # 是否显示进度
                return_tokens=USE_DIFFUSION  # 是否返回令牌，取决于是否使用扩散模型
            )
        else:  # 如果没有旋律
            outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)  # 仅使用文本生成音乐
    except RuntimeError as e:  # 捕获运行时错误
        raise gr.Error("Error while generating " + e.args[0])  # 抛出 Gradio 错误，显示生成过程中的错误信息
    if USE_DIFFUSION:  # 如果使用 MultiBandDiffusion
        if gradio_progress is not None:  # 如果提供了 Gradio 进度条
            gradio_progress(1, desc='Running MultiBandDiffusion...')  # 更新进度条，显示正在运行 MultiBandDiffusion
        tokens = outputs[1]  # 获取生成的令牌
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):  # 如果模型是立体声压缩模型
            left, right = MODEL.compression_model.get_left_right_codes(tokens)  # 获取左右声道的令牌
            tokens = torch.cat([left, right])  # 拼接左右声道令牌
        outputs_diffusion = MBD.tokens_to_wav(tokens)  # 使用 MultiBandDiffusion 将令牌转换为波形
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):  # 如果模型是立体声压缩模型
            assert outputs_diffusion.shape[1] == 1  # 确保输出是单声道
            outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)  # 重排输出为立体声格式
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)  # 拼接原始输出和扩散输出
    outputs = outputs.detach().cpu().float()  # 将输出从 GPU 转移到 CPU，并转换为浮点型
    pending_videos = []  # 初始化待处理的视频列表
    out_wavs = []  # 初始化输出 WAV 文件列表
    for output in outputs:  # 遍历生成的输出
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:  # 创建临时 WAV 文件
            audio_write(  # 写入音频数据到文件
                file.name, output, MODEL.sample_rate, strategy="loudness",  # 文件名、音频数据、采样率和音量策略
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)  # 音量参数设置
            pending_videos.append(pool.submit(make_waveform, file.name))  # 提交任务到进程池，生成波形视频
            out_wavs.append(file.name)  # 添加 WAV 文件路径到列表
            file_cleaner.add(file.name)  # 将文件添加到清理器
    out_videos = [pending_video.result() for pending_video in pending_videos]  # 获取所有生成的视频路径
    for video in out_videos:  # 遍历生成的视频
        file_cleaner.add(video)  # 将视频文件添加到清理器
    print("batch finished", len(texts), time.time() - be)  # 打印批次完成信息和耗时
    print("Tempfiles currently stored: ", len(file_cleaner.files))  # 打印当前存储的临时文件数量
    return out_videos, out_wavs  # 返回生成的视频和 WAV 文件路径


def predict_batched(texts, melodies):  # 定义 predict_batched 函数，用于批量预测
    max_text_length = 512  # 定义最大文本长度为 512 字符
    texts = [text[:max_text_length] for text in texts]  # 截断文本到最大长度
    load_model('facebook/musicgen-stereo-melody')  # 加载立体声旋律模型
    res = _do_predictions(texts, melodies, BATCHED_DURATION)  # 调用 _do_predictions 函数进行预测，设置时长为批量模式时长
    return res  # 返回预测结果


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):  # 定义 predict_full 函数，用于完整预测
    global INTERRUPTING  # 声明使用全局变量 INTERRUPTING
    global USE_DIFFUSION  # 声明使用全局变量 USE_DIFFUSION
    INTERRUPTING = False  # 重置中断标志为 False
    progress(0, desc="Loading model...")  # 更新进度条，显示正在加载模型
    model_path = model_path.strip()  # 去除模型路径两端的空白字符
    if model_path:  # 如果提供了模型路径
        if not Path(model_path).exists():  # 如果路径不存在
            raise gr.Error(f"Model path {model_path} doesn't exist.")  # 抛出错误，显示路径不存在
        if not Path(model_path).is_dir():  # 如果路径不是目录
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")  # 抛出错误，显示路径必须是包含特定文件的目录
        model = model_path  # 将模型设置为提供的路径
    if temperature < 0:  # 如果温度小于 0
        raise gr.Error("Temperature must be >= 0.")  # 抛出错误，显示温度必须大于等于 0
    if topk < 0:  # 如果 Top-k 小于 0
        raise gr.Error("Topk must be non-negative.")  # 抛出错误，显示 Top-k 必须是非负数
    if topp < 0:  # 如果 Top-p 小于 0
        raise gr.Error("Topp must be non-negative.")  # 抛出错误，显示 Top-p 必须是非负数

    topk = int(topk)  # 将 Top-k 转换为整数
    if decoder == "MultiBand_Diffusion":  # 如果解码器选择为 MultiBand_Diffusion
        USE_DIFFUSION = True  # 设置使用扩散模型标志为 True
        progress(0, desc="Loading diffusion model...")  # 更新进度条，显示正在加载扩散模型
        load_diffusion()  # 加载扩散模型
    else:  # 如果选择其他解码器
        USE_DIFFUSION = False  # 设置使用扩散模型标志为 False
    load_model(model)  # 加载指定的模型

    max_generated = 0  # 初始化最大已生成量为 0

    def _progress(generated, to_generate):  # 定义 _progress 函数，用于更新生成进度
        nonlocal max_generated  # 声明使用非局部变量 max_generated
        max_generated = max(generated, max_generated)  # 更新最大已生成量
        progress((min(max_generated, to_generate), to_generate))  # 更新进度条，显示当前生成量和总生成量
        if INTERRUPTING:  # 如果中断标志为 True
            raise gr.Error("Interrupted.")  # 抛出错误，显示已中断
    MODEL.set_custom_progress_callback(_progress)  # 设置模型的自定义进度回调函数

    videos, wavs = _do_predictions(  # 调用 _do_predictions 函数进行预测
        [text], [melody], duration, progress=True,  # 传递文本、旋律、时长和进度标志
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,  # 传递生成参数
        gradio_progress=progress)  # 传递 Gradio 进度条
    if USE_DIFFUSION:  # 如果使用扩散模型
        return videos[0], wavs[0], videos[1], wavs[1]  # 返回原始和扩散模型生成的视频和音频
    return videos[0], wavs[0], None, None  # 返回生成的视频和音频，扩散模型输出为 None


def toggle_audio_src(choice):  # 定义 toggle_audio_src 函数，用于切换音频输入源
    if choice == "mic":  # 如果选择为麦克风
        return gr.update(source="microphone", value=None, label="Microphone")  # 更新音频输入为麦克风
    else:  # 如果选择为文件
        return gr.update(source="upload", value=None, label="File")  # 更新音频输入为文件上传


def toggle_diffusion(choice):  # 定义 toggle_diffusion 函数，用于切换解码器选项
    if choice == "MultiBand_Diffusion":  # 如果选择为 MultiBand_Diffusion
        return [gr.update(visible=True)] * 2  # 返回两个可见性更新，显示扩散模型输出
    else:  # 如果选择其他解码器
        return [gr.update(visible=False)] * 2  # 返回两个可见性更新，隐藏扩散模型输出


def ui_full(launch_kwargs):  # 定义 ui_full 函数，用于创建完整的 Gradio 界面
    with gr.Blocks() as interface:  # 使用 Gradio Blocks 创建界面
        gr.Markdown(  # 添加 Markdown 文本，显示标题和描述
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():  # 创建一行布局
            with gr.Column():  # 创建一列布局
                with gr.Row():  # 创建嵌套行布局
                    text = gr.Text(label="Input Text", interactive=True)  # 创建文本输入框，用于输入音乐描述
                    with gr.Column():  # 创建嵌套列布局
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")  # 创建单选按钮，用于选择旋律输入方式
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")  # 创建音频输入组件，初始为文件上传
                with gr.Row():  # 创建嵌套行布局
                    submit = gr.Button("Submit")  # 创建提交按钮，用于触发生成
                    # 改编自 https://github.com/rkfg/audiocraft/blob/long/app.py, MIT 许可证
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)  # 创建中断按钮，点击时调用 interrupt 函数
                with gr.Row():  # 创建嵌套行布局
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                      "facebook/musicgen-large", "facebook/musicgen-melody-large",
                                      "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                                      "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                                      "facebook/musicgen-stereo-melody-large"],
                                     label="Model", value="facebook/musicgen-stereo-melody", interactive=True)  # 创建模型选择单选按钮
                    model_path = gr.Text(label="Model Path (custom models)")  # 创建文本输入框，用于自定义模型路径
                with gr.Row():  # 创建嵌套行布局
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                       label="Decoder", value="Default", interactive=True)  # 创建解码器选择单选按钮
                with gr.Row():  # 创建嵌套行布局
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)  # 创建时长滑块
                with gr.Row():  # 创建嵌套行布局
                    topk = gr.Number(label="Top-k", value=250, interactive=True)  # 创建 Top-k 数值输入框
                    topp = gr.Number(label="Top-p", value=0, interactive=True)  # 创建 Top-p 数值输入框
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)  # 创建温度数值输入框
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)  # 创建分类器自由引导数值输入框
            with gr.Column():  # 创建另一列布局
                output = gr.Video(label="Generated Music")  # 创建视频输出组件，用于显示生成的音乐视频
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')  # 创建音频输出组件，用于显示生成的 WAV 文件
                diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")  # 创建视频输出组件，用于显示 MultiBand Diffusion 解码器的输出
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')  # 创建音频输出组件，用于显示 MultiBand Diffusion 解码器的 WAV 文件
        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, model_path, decoder, text, melody, duration, topk, topp,
                                                                     temperature, cfg_coef],
                                               outputs=[output, audio_output, diffusion_output, audio_diffusion])  # 提交按钮点击事件：先切换解码器可见性，然后调用 predict_full 函数
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)  # 单选按钮变化事件：切换音频输入源

        gr.Examples(  # 创建示例组件，展示预定义的输入示例
            fn=predict_full,  # 示例调用的函数
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default"
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default"
                ],
                [
                    "Punk rock with loud drum and power guitar",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "MultiBand_Diffusion"
                ],
            ],
            inputs=[text, melody, model, decoder],  # 示例输入组件
            outputs=[output]  # 示例输出组件
        )
        gr.Markdown(  # 添加 Markdown 文本，显示更多详细信息
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass.

            The model was trained with description from a stock music catalog, descriptions that will work best
            should include some level of details on the instruments present, along with some intended use case
            (e.g. adding "perfect for a commercial" can somehow help).

            Using one of the `melody` model (e.g. `musicgen-melody-*`), you can optionally provide a reference audio
            from which a broad melody will be extracted.
            The model will then try to follow both the description and melody provided.
            For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

            It is now possible to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min).
            An overlap of 12 seconds is kept with the previously generated chunk, and 18 "new" seconds
            are generated each time.

            We present 10 model variations:
            1. facebook/musicgen-melody -- a music generation model capable of generating music condition
                on text and melody inputs. **Note**, you can also use text only.
            2. facebook/musicgen-small -- a 300M transformer decoder conditioned on text only.
            3. facebook/musicgen-medium -- a 1.5B transformer decoder conditioned on text only.
            4. facebook/musicgen-large -- a 3.3B transformer decoder conditioned on text only.
            5. facebook/musicgen-melody-large -- a 3.3B transformer decoder conditioned on and melody.
            6. facebook/musicgen-stereo-*: same as the previous models but fine tuned to output stereo audio.

            We also present two way of decoding the audio tokens
            1. Use the default GAN based compression model. It can suffer from artifacts especially
                for crashes, snares etc.
            2. Use [MultiBand Diffusion](https://arxiv.org/abs/2308.02560). Should improve the audio quality,
                at an extra computational cost. When this is selected, we provide both the GAN based decoded
                audio, and the one obtained with MBD.

            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)  # 启动 Gradio 界面，应用启动参数


def ui_batched(launch_kwargs):  # 定义 ui_batched 函数，用于创建批量处理的 Gradio 界面
    with gr.Blocks() as demo:  # 使用 Gradio Blocks 创建界面
        gr.Markdown(  # 添加 Markdown 文本，显示标题和描述
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():  # 创建一行布局
            with gr.Column():  # 创建一列布局
                with gr.Row():  # 创建嵌套行布局
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)  # 创建文本输入框，用于描述音乐
                    with gr.Column():  # 创建嵌套列布局
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")  # 创建单选按钮，用于选择旋律输入方式
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")  # 创建音频输入组件，初始为文件上传
                with gr.Row():  # 创建嵌套行布局
                    submit = gr.Button("Generate")  # 创建生成按钮，用于触发预测
            with gr.Column():  # 创建另一列布局
                output = gr.Video(label="Generated Music")  # 创建视频输出组件，用于显示生成的音乐视频
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')  # 创建音频输出组件，用于显示生成的 WAV 文件
        submit.click(predict_batched, inputs=[text, melody],
                     outputs=[output, audio_output], batch=True, max_batch_size=MAX_BATCH_SIZE)  # 提交按钮点击事件：调用 predict_batched 函数进行批量预测
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)  # 单选按钮变化事件：切换音频输入源
        gr.Examples(  # 创建示例组件，展示预定义的输入示例
            fn=predict_batched,  # 示例调用的函数
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],  # 示例输入组件
            outputs=[output]  # 示例输出组件
        )
        gr.Markdown("""
        ### More details

        The model will generate 15 seconds of audio based on the description you provided.
        The model was trained with description from a stock music catalog, descriptions that will work best
        should include some level of details on the instruments present, along with some intended use case
        (e.g. adding "perfect for a commercial" can somehow help).

        You can optionally provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

        You can access more control (longer generation, more models etc.) by clicking
        the <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        (you will then need a paid GPU from HuggingFace).
        If you have a GPU, you can run the gradio demo locally (click the link to our repo below for more info).
        Finally, you can get a GPU for free from Google
        and run the demo in [a Google Colab.](https://ai.honu.io/red/musicgen-colab).

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
        for more details. All samples are generated with the `stereo-melody` model.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)  # 启动 Gradio 界面，设置最大队列大小并应用启动参数


if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 对象，用于解析命令行参数
    parser.add_argument(  # 添加命令行参数 --listen
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(  # 添加命令行参数 --username
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(  # 添加命令行参数 --password
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(  # 添加命令行参数 --server_port
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(  # 添加命令行参数 --inbrowser
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(  # 添加命令行参数 --share
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()  # 解析命令行参数

    launch_kwargs = {}  # 初始化启动参数字典
    launch_kwargs['server_name'] = args.listen  # 设置服务器名称（监听 IP）

    if args.username and args.password:  # 如果提供了用户名和密码
        launch_kwargs['auth'] = (args.username, args.password)  # 设置认证参数
    if args.server_port:  # 如果提供了服务器端口
        launch_kwargs['server_port'] = args.server_port  # 设置服务器端口
    if args.inbrowser:  # 如果设置了在浏览器中打开
        launch_kwargs['inbrowser'] = args.inbrowser  # 设置在浏览器中打开参数
    if args.share:  # 如果设置了共享 UI
        launch_kwargs['share'] = args.share  # 设置共享参数

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # 配置日志记录，设置为 INFO 级别，输出到 stderr

    # 显示界面
    if IS_BATCHED:  # 如果是批量模式
        global USE_DIFFUSION  # 声明使用全局变量 USE_DIFFUSION
        USE_DIFFUSION = False  # 设置不使用扩散模型
        ui_batched(launch_kwargs)  # 调用批量界面函数
    else:  # 如果不是批量模式
        ui_full(launch_kwargs)  # 调用完整界面函数
