#!/bin/bash

echo "---"
echo "Main container started."
echo "Waiting 10 seconds for the nginx sidecar to boot up..."
echo "---"

# (amlt 应该会等待端口就绪, 但我们为了保险起见多等10秒)
sleep 10

echo "Attempting to connect to sidecar at http://localhost:80"

# -v (verbose) 显示连接详情
# --fail (失败时退出)
curl -v --fail http://localhost:80

# $? 检查上一个命令的退出码 (0 = 成功)
if [ $? -eq 0 ]; then
  echo "---"
  echo "✅ ✅ ✅ SUCCESS! ✅ ✅ ✅"
  echo "Sidecar is running and accessible from the main container."
  echo "你的集群支持 'resources.properties.sidecars' 语法！"
  echo "---"
else
  echo "---"
  echo "❌ ❌ ❌ FAILURE! ❌ ❌ ❌"
  echo "Could not connect to the sidecar at localhost:80."
  echo "你的集群或amlt设置不支持 'resources.properties.sidecars' 语法。"
  echo "---"
  exit 1 # 让 AML 作业失败
fi