# yyts后端

这是nw `yyts` AgentFlow 使用的本地媒体执行后端。

## 从 GitHub 部署

在 Windows、PowerShell、Git、Python 3.12、Python 3.10、NVIDIA 驱动/CUDA 和
FFmpeg 已准备好的前提下：

```powershell
git clone <仓库地址>
cd 译影同声
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\start_backend.ps1
```

`setup.ps1` 使用锁定文件和 `backend/models.lock.json` 中的 revision，创建
两个独立 Python 环境并下载 Whisper、Demucs、CosyVoice。模型、任务文件和
运行环境不会提交到 GitHub。

启动后检查 `http://127.0.0.1:5000/api/health`。需要让女娲云端访问时，另开
PowerShell 窗口运行：

```powershell
cloudflared tunnel --url http://127.0.0.1:5000
```

将生成的 HTTPS 地址配置到 AgentFlow 的 HTTP 节点。任意时刻只启动一台机器
的后端，另一台保持停止状态。
