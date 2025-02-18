# 使用官方的 Python 镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的 /app 目录下
COPY . /app

#安装c++编译器
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装所需的 Python 包
RUN pip install nano-graphrag

# # 设置容器启动时执行的命令
# CMD ["python", "your_script.py"]
CMD ["tail", "-f", "/dev/null"]