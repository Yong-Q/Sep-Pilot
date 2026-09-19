#!/bin/bash
#PBS -N jupyter
#PBS -l nodes=zx01:ppn=49
#PBS -o log.log
#PBS -e log.err

#input_path="./input/"
# 并行进程数
num_processes=48
# 超时时间（秒）
timeout=70000

# 初始化计数器
processed=0
successful=0
#source ee.sh
# 处理函数，接收输入文件路径作为参数
process_input() {
      # 提取输入文件名
   # echo $1
    input_file=$(basename "$1")
    echo "Processing ${input_file}..."
            if [ ! -f "$1" ]; then
                    echo "输入文件 $1 不存在"
                    break  # 如果不存在，跳出循环
                    fi
              # 构建输入目录并拷贝文件
                input_dir="./input_COF/$input_file"
                  mkdir -p "$input_dir"
                  mkdir -p "out_COF"
                  output_file="out_COF"
                    cp "$1" "$input_dir/input.dat" 
                      
                        # 进入输入目录并启动 cof 命令
                          pushd "$input_dir" > /dev/null
                           # echo *
                          #    command & # 运行其他命令
                                ../../ceshi1 &   # 将 my 命令放到后台运行
                                  pid=$! # 记录 my 命令的 PID
                                    
                                      # 等待 my 命令执行完成或超时
                                        start=$(date +%s)
                                          while ps -p $pid > /dev/null; do
                                                  now=$(date +%s)
                                                      elapsed=$((now-start))
                                                          if [ $elapsed -gt $timeout ]; then
                                                                    echo "Iteration $input_file timeout, skip."
                                                                          kill $pid 
                                                                                break
                                                                                    fi
                                                                                      #  sleep 1
                                                                                          done
                                                                                            
                                                                                              # 检查输出文件并拷贝
                                                                                                if grep -q "Molecule" output.dat && ! ps -p $pid > /dev/null; then  
                                                                                                        cp output.dat ../../$output_file/"$input_file"
                                                                                                        rm  Vext*
                                                                                                        rm density*
                                                                                                            ((successful+=1))
                                                                                                                echo "${input_file} processed successfully."
                                                                                                                  else
                                                                                                                          echo "${input_file} processing failed."
                                                                                                                            fi
                                                                                                                              
                                                                                                                                # 恢复上级目录
                                                                                                                                  popd > /dev/null 
                                                                                                                                    
                                                                                                                                      # 更新计数器
                                                                                                                                        ((processed+=1)) 
}
#input="./input/1.dat"
#find "$input_path" -type f -iname "*.dat" -print
# 寻找所有输入文件，并使用 xargs 并行处理
#bash -c './co.sh;process_input "./input/input.dat"'
#find "$input_path" -type f -print0 | xargs -0 -P $num_processes bash -c 'process_input "$@
#find "$input_path" -type f -name '*.dat' -print0 | xargs -0 -P $num_processes -I{} sh -c 'process_input {} &
#echo "All ${processed} inputs are processed. ${successful} processed successfully."
