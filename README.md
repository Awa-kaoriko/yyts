# yyts后端

这是nw `yyts` AgentFlow 使用的本地媒体执行后端。

## 准备FFmpeg
powershell -winget install --id Gyan.FFmpeg.Shared -e

安装后关闭并重新打开 PowerShell


看到：Pinned environments and models are ready.才说明安装完成。然后运行：
powershell -ExecutionPolicy Bypass -File .\start_backend.ps1


## 从 GitHub 部署

在 Windows、PowerShell、Git、Python 3.12、Python 3.10、NVIDIA 驱动/CUDA 和
FFmpeg 已准备好的前提下：

```powershell
git clone https://github.com/Awa-kaoriko/yyts.git
cd yyts
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
setup.ps1 会自动完成：
创建 backend\.venv（Python 3.12）
创建 backend\runtime\cosyvoice_env（Python 3.10）
安装锁定依赖
下载固定版本 Whisper
下载固定版本 Demucs
下载固定版本 CosyVoice
克隆固定版本 CosyVoice 源码
自动配置项目内模型和临时目录

`setup.ps1` 使用锁定文件和 `backend/models.lock.json` 中的 revision，创建
两个独立 Python 环境并下载 Whisper、Demucs、CosyVoice。

## 启动后端
```powershell
powershell -ExecutionPolicy Bypass -File .\start_backend.ps1
```

启动后检查 `http://127.0.0.1:5000/api/health`。需要让nw云端访问时，另开
PowerShell 窗口运行：

```powershell
cloudflared tunnel --url http://127.0.0.1:5000
```

将生成的 HTTPS 地址配置到 AgentFlow 的 HTTP 节点。任意时刻只启动一台机器
的后端，另一台保持停止状态。
