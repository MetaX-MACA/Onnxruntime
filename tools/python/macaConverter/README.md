--------------------------------------------------------------------------------------------
(下面的命令都是以源码方式运行的，如果需要以whl安装包的方式运行，只需要将命令中的model_convert.py改为 -m maca_converter，如python -m maca_converter --model_path ./caffe_model --model_type caffe --output ./output.onnx)
---------------------------------------------------------------------------------------------
1 caffe转onnx
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx
   参数说明：model_path：caffe模型所在的文件夹，文件夹里需要有对应的.caffemodel文件和.prototxt文件
                   model_type：模型类型，此处固定为caffe
                   output：输出onnx模型的文件路径
   
2 tensorflow(h5)转onnx
   命令：python model_convert.py --model_path ./test.h5 --model_type tf-h5  --output ./output.onnx
   参数说明：model_path：h5模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为tf-h5
                   output：输出onnx模型的文件路径  

3 tensorflow(savemodel)转onnx
   命令：python model_convert.py --model_path ./tfsm_model  --model_type tf-sm --output ./output.onnx
   参数说明：model_path：savemodel模型所在的文件夹，文件夹里需要有对应的assets(文件夹)/saved_model.pb(文件)/variables(文件夹)
                   model_type：模型类型，此处固定为tf-sm
                   output：输出onnx模型的文件路径

4 tensorflow(checkpoint)转onnx
   命令：python model_convert.py --model_path ./ckpt_models/test.meta --model_type tf-ckpt  --output ./output.onnx --inputs x:0,y:0  --outputs op_to_store:0
   参数说明：model_path：checkpoint .meta文件所在的路径
                   model_type：模型类型，此处固定为tf-ckpt
                   output：输出onnx模型的文件路径
                   inputs：原始模型的输入变量名称
                   outputs：原始模型的输出变量名称

5 tensorflow(graphpb)转onnx
   命令：python model_convert.py --model_path ./graph.pb  --model_type tf-graph  --output ./output.onnx --inputs x:0,y:0  --outputs op_to_store:0
   参数说明：model_path：graph模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为tf-graph
                   output：输出onnx模型的文件路径
                   inputs：原始模型的输入变量名称
                   outputs：原始模型的输出变量名称

6 pytorch转onnx
   命令： python model_convert.py --model_path ./mnist_model.pkl  --model_type pytorch  --output ./torch.onnx  --model_def_file  ./CNN.py  --model_class_name CNN --input_shape [1,1,28,28]   (完整模型加权重参数)
              或 python model_convert.py --model_path ./mnist_model.pkl  --model_type pytorch  --output ./unet.onnx  --model_def_file  ./unet.py   --model_class_name Net  --model_weights_file ./9_epoch_iou_0.9743422508239746.pth  --input_shape [64,3,32,32]  (只含权重参数，model_path可任意指定，实际不会使用)
              或 python model_convert.py --model_path ./xxx --model_type pytorch  --output ./resnet50.onnx  --model_class_name torchvision.models.resnet50  --model_weights_file ./0.96966957919051920.9139525532770406.pth   --input_shape [16,3,256,256]  (只含权重参数，使用torchvision中的类，model_path可任意指定，实际不会使用，不需要model_def_file参数)
   参数说明：model_path：pytorch模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为pytorch
                   output：输出onnx模型的文件路径
                   input shape：输入shape
                   model_def_file：模型定义文件
                   model_weights_file：权重文件
                   model_class_name：类名(可以是自定义的类或pytorch提供的模型类)

   多输入/多输出模型：
     python model_convert.py --model_path ./multi_input.pth  --model_type pytorch  --output ./torch.onnx  --model_def_file  ./pt_multi_input.py   --model_class_name nettest --input_shape [1,3,200,300]/[1,3,200,300] --output_num 2
   或：
     python model_convert.py --model_path ./xxx   --model_type pytorch  --output ./torch.onnx  --model_def_file  ./pt_multi_input.py   --model_class_name nettest  --model_weights_file ./multi_input_state.pth  --input_shape [1,3,500,600]/[1,3,500,600] --output_num 2  

   如类的定义中需要输入参数，参考以下方式(--params_file指定参数定义的文件位置)：
     python model_convert.py --model_path ./xxxx   --model_type pytorch  --output ./torch.onnx  --model_def_file  ./vggnet.py   --model_class_name VGGbase  --model_weights_file ./vggnet_params.pth  --input_shape [1,3,28,28] --params_file ./vgg_params.py 
   (参数定义文件中，需以字典形式存储，且字典名称固定为param_dict, 字典键值必须与类中需要的初始化变量名称一致，可从sample_models中下载样例)      

7 darknet转onnx
   命令：python model_convert.py --model_path ./dn_models --model_type darknet  --output ./output.onnx
   参数说明：model_path：darknet模型所在的文件夹，文件夹里需要有对应的.cfg文件和.weights文件
                   model_type：模型类型，此处固定为darknet
                   output：输出onnx模型的文件路径

8 paddle转onnx
   命令：
          (1)动态paddle模型
            python model_convert.py --model_path ./xxx   --model_type paddle   --output ./paddle.onnx  --model_def_file  ./mnist.py --model_class_name LeNet  --model_weights_file ./paddle_checkpoint/final.pdparams --input_shape [1,1,28,28]  (自定义的类，在mnist.py中实现class LeNet)
          或python model_convert.py --model_path ./xxx  --model_type paddle   --output ./paddle.onnx  --model_class_name paddle.vision.models.LeNet  --model_weights_file ./paddle_checkpoint/final.pdparams --input_shape [1,1,28,28]    (paddle自带的类，类名为LeNet)
        (2)静态paddle模型
          python model_convert.py --model_path ./paddle_model   --model_type paddle   --output ./paddle.onnx 
           
   参数说明：model_path：pytorch模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为paddle
                   output：输出onnx模型的文件路径
                   (以下五个参数，动态paddle模型才需要输入)
                   input shape：输入shape
                   model_def_file：模型定义文件
                   model_weights_file：模型权重文件
                   model_class_name：类名(可以是自定义的类或paddle提供的模型类)
                   paddle_input_type：输入数据类型(可不指定，默认为float32)
                   (动态paddle模型，对应的model_path字段，可任意填写(如：--model_path ./xxx )，实际不会用到)
                   (如果是动态模型且调用paddle自带的模型类(如paddle.vision.models.LeNet，则不需要指定model_def_file参数))
   多输入/多输出模型：
     python model_convert.py --model_path ./xxx   --model_type paddle  --output ./paddle.onnx  --model_def_file  ./paddle_mi.py  --model_class_name MyModel  --model_weights_file ./mi.pdparams --input_shape [1,3,200,300]/[1,3,200,300]

****************************************************************************************
9 启用动态batch(默认关闭)
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx  --dynamic_batch 1
                
10 关闭simplify功能(默认打开)
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx  --simplify 0
   参数说明：
            如果模型为动态batch，经优化后的模型欲修改为静态batch(默认batch为1)，可指定参数 --simplify 2.
            如果模型为动态shape，可通过参数--simplify 2 --simplify_hw 256,256 指定维度生成模型，
            也可不带参数，优化后的模型仍保持为动态shape(不会做常量折叠的优化操作)。

11 启用fp32-->fp16转换(默认关闭)
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx  --fp32_to_fp16 1
   如原始模型为qdq量化后的模型，执行simplify时可能失败，关闭simplify功能即可：python model_convert.py --model_path ./qdq.onnx --model_type onnx --output ./fp16.onnx  --fp32_to_fp16 1 --simplify 0

12 提取子图
   命令：python model_convert.py --model_path ./output.onnx  --model_type onnx   --output ./test.onnx  --extract_sub 1 --inputs input_1:0  --outputs functional_1/concatenate/concat:0
   说明：model_type必须为onnx

13 op_set版本转换
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx  --op_set 12
   也可直接对onnx模型进行版本转换：python model_convert.py --model_path ./my.onnx --model_type onnx --output ./test.onnx  --op_set 12

14 pad+pool融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx  --fuse_pad_pool 1
   或：python model_convert.py --model_path ./test.h5 --model_type tf-h5  --output ./output.onnx --fuse_pad_pool 1
   
15 BN转Conv(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --bn_to_conv 1
   说明: 支持将无法融合的BN算子转换成1x1的分组卷积

16 Resize算子合成(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --expand_to_resize 1
   说明: 支持将Reshape+Expand+Reshape算子转换成Resize算子        

17 Gelu融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_gelu 1
   说明: 支持将以下两种算子组合转换为Gelu算子：
         1.Pow+Mul+Add+Mul+Tanh+Add+Mul+Mul
         2.Div+Erf+Add+Mul+Mul

18 输入类型修改(float-->uint8，默认关闭)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fp32_to_u8 1
   说明: 支持将模型的输入类型从float转为uint8

19 LayerNorm转换(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_layernorm 1
   说明: 支持将匹配的算子组合转换成LayerNorm算子

20 MatMul+Add转Gemm(默认关闭)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --matmul_to_gemm 1
   说明: 支持将MatMul(A、B维度均为2)+Add算子组合转换成Gemm算子

21 mish合成(默认开启)
   命令：python model_convert.py --model_path ./dn_models --model_type darknet  --output ./output.onnx --support_mish 1
   也可直接对onnx模型中的算子进行合成：python model_convert.py --model_path ./my.onnx --model_type onnx --output ./test.onnx  --support_mish 1

22 swish合成(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --support_swish 1
   说明: 仅支持model_type为onnx, 调用命令会自动将模型中符合条件的Sigmoid+Mul组合转换为Swish(或将HardSigmoid+Mul(或Add+Clip+Mul+Div)组合转换为HardSwish)

23 HardSigmoid融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_hard_sigmoid 1
   说明: 支持将Add+Clip+Div组合转换为HardSigmoid算子(需满足: 1.Add算子的input[1]==3; 2.Clip算子的min=0,max=6;3.Div算子的input[1]=6)

24 ReduceL2融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx  --fuse_reducel2 1
   说明: 支持将以下三种算子组合融合为ReduceL2算子：
      1.Pow-->ReduceSum-->Add-->Sqrt
      2.Pow-->ReduceSum-->Sqrt
      3.Mul-->ReduceSum-->Sqrt

25 mha融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_mha 1
   说明: 支持将KQV组合融合成MultiHeadAttentionV1算子

26 ConvTranpose+Add+BN融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_ct_add_bn 1
   说明: 支持将ConvTranpose+Add+BN或ConvTranpose+Add融合进ConvTranpose算子

27 MatMul+Reshape+BN+Reshape融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_matmul_bn 1
   说明: 支持将MatMul+Reshape+BN+Reshape融合进Gemm算子

28 Conv+Add/Sub/Mul/Div融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_conv_asmd 1
   说明: 当Add/Sub/Mul/Div中的操作数为常量时，支持将该算子融合进与其相邻的Conv算子(如Conv+Add、Conv+Sub、Conv+Mul、Conv+Div)

29 simplify优化是否跳过常量折叠(默认关闭)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --simplify_skip_constant_folding 1
   说明：在使用 onnxsim.simplify 进行优化时，支持外部配置是否进行常量折叠,默认进行常量折叠

30 gemm+bn融合(默认开启)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --fuse_gemm_bn 0
   说明：支持将bn融合进gemm算子

31 simplify优化是否跳过optimization优化(默认关闭)
   命令：python model_convert.py --model_path ./test.onnx --model_type onnx --output ./output.onnx --simplify_skip_optimization 1
   说明：在使用 onnxsim.simplify 进行优化时，支持外部配置是否进行optimization优化,默认进行optimization优化
