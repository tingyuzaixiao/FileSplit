import subprocess
import os
import sys
from pathlib import Path


def convert_pdf_with_mineru(pdf_path: str,
                            output_dir: str,
                            gpu_memory_utilization: float=0.2,
                            timeout: float=600.0,
                            conda_env="mineru-env"):
    pdf_path = os.path.abspath(pdf_path)
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)

    # 构建 conda run 命令（自动处理环境变量和依赖）
    cmd = [
        "conda", "run", "-n", conda_env,
        "mineru", "-p", pdf_path, "-o", output_path,
        "--gpu-memory-utilization", gpu_memory_utilization
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8' if sys.platform != 'win32' else None
        )

        if result.returncode == 0:
            file_name_no_ext = Path(pdf_path).stem
            dest_file_name = file_name_no_ext + ".md"
            dest_file_path = Path(output_path) / file_name_no_ext / "hybrid_auto" / dest_file_name
            return True, str(dest_file_path)
        else:
            return False, f"转换失败:\n{result.stderr}"

    except FileNotFoundError:
        return False, "错误：未找到 'conda' 命令。请确保 Conda 已加入系统 PATH，或使用绝对路径调用。"
    except subprocess.TimeoutExpired:
        return False, "转换超时，请检查 PDF 大小或增加 timeout 参数。"
    except Exception as e:
        return False, f"发生异常: {str(e)}"


# 使用示例
if __name__ == "__main__":
    success, msg = convert_pdf_with_mineru(
        pdf_path="/home/zhangjiang/广东大湾区空天信息研究院2025年度职工考核评价办法.pdf",
        output_dir="/home/zhangjiang/output",
        conda_env="mineru-env",
        gpu_memory_utilization=0.2
    )
    print(success)
    print(msg)