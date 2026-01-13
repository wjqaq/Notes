方法1：
- 使用命令wsl --install -d Ubuntu-24.04 --name Ubuntu-24.04 --location "E:\WSL\Ubuntu"，即可下载指定的发行版本到指定的路径下。
这个方法是便利但是可能会下载很慢（需要魔法）。
方法2：
- 先下载一个WSL发行版本，然后通过命令wsl --import Ubuntu2204 "E:\WSL\Ubuntu2204" "Ubuntu2204.tar.gz" --version 2，安装到指定路径下。
