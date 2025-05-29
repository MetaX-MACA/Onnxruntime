# macaPrecision  <br>
功能：<br> 
---- 
    调试ONNX模型，自动加载运行模型两次(cpu<--->gpu 或 cpu<--->acuity 或 gpu<--->acuity)，对比不同工具执行结果，cpu与gpu不一致时输出错误节点的名称及输入信息；cpu/gpu与acuity则会输出所有节点的对比信息（无论是否一致）。
    注意：使用acuity模式时需预先安装好acuity-tool.

安装：<br>
---
    pip install maca_precision-x.x.x-py3-none-any.whl

运行方式：<br> 
--------  
    python onnx_debug.py -i ./test.onnx -m CPU-MACA 或
    python onnx_debug.py -f ./test_matmul -m CPU-MACA 或
    python onnx_debug.py -i ./test.onnx -m CPU-Acuity 或
    python onnx_debug.py -i ./test.onnx -m CPU-Acuity -q
    python onnx_debug.py -i ./test.onnx -m MACA-Acuity 或
    python onnx_debug.py -i ./test.onnx -m MACA-Acuity -q

运行参数：<br>
--------
    -i xxxxx.onnx 输入onnx模型路径
    -f xxxxx 测试模型所在文件夹路径
    -m CPU-Acuity/MACA-Acuity/CPU-MACA精度对比模式，默认为CPU-MACA
    -c snr/mse/cosine 精度测试方法，默认为snr
    -q 当测试模型为量化后模型，指定此参数时，只输出量化节点精度对比信息，其他情况不生效

当前限制： <br>
---------                                                                                                                                                                                                                                              
     暂未在gpu环境测试，只在cpu环境下模拟了output不一致的场景，如需测试gpu，请在代码中搜索EP_list进行替换。  

输出：
------------ 
    CPU VS GPU                                                                                                                                                
     1 结果不一致                                                                         
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                  
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                                                            
      WARNING: output  2  is abnormal, please check it~~                                                                                          
      Dismatch node name:  MatMul_1 , input: ['0', '1']                                  
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                                                                
                                                                                                                                                        
     2 结果一致                                  
      =================================================                                             
      =================================================                                                                                
      Congratulations, it works well as expected        

    CPU/GPU VS Acuity
     
        I:[ONNXOPTIMIZER]:Acuity compare with ORT CPUEP:
        Acuity: Add_Add_103_36_out0_nchw_1_2048_7_7.qnt.tensor                                   ORT-CPU: Add_103->PPQ_Variable_1448     snr:0.00741552393426903
        Acuity: Add_Add_110_22_out0_nchw_1_2048_7_7.qnt.tensor                                   ORT-CPU: Add_110->PPQ_Variable_1532     snr:0.0013733154811174971
        Acuity: Add_Add_117_10_out0_nchw_1_2048_7_7.qnt.tensor                                   ORT-CPU: Add_117->PPQ_Variable_1616     snr:0.08514950590065112
        Acuity: Add_Add_16_216_out0_nchw_1_256_56_56.qnt.tensor                                  ORT-CPU: Add_16->PPQ_Variable_422       snr:0.000703290732665197
        Acuity: Add_Add_23_201_out0_nchw_1_256_56_56.qnt.tensor                                  ORT-CPU: Add_23->PPQ_Variable_506       snr:0.04591139450490403
        Acuity: Add_Add_31_186_out0_nchw_1_512_28_28.qnt.tensor                                  ORT-CPU: Add_31->PPQ_Variable_596       snr:0.0013660383248182697
        Acuity: Add_Add_38_171_out0_nchw_1_512_28_28.qnt.tensor                                  ORT-CPU: Add_38->PPQ_Variable_680       snr:0.0027642061539056
        Acuity: Add_Add_45_156_out0_nchw_1_512_28_28.qnt.tensor                                  ORT-CPU: Add_45->PPQ_Variable_764       snr:0.006850498242534273
        Acuity: Add_Add_52_141_out0_nchw_1_512_28_28.qnt.tensor                                  ORT-CPU: Add_52->PPQ_Variable_848       snr:0.15184437447830884
        Acuity: Add_Add_60_126_out0_nchw_1_1024_14_14.qnt.tensor                                 ORT-CPU: Add_60->PPQ_Variable_938       snr:0.0029793715924434938
        Acuity: Add_Add_67_111_out0_nchw_1_1024_14_14.qnt.tensor                                 ORT-CPU: Add_67->PPQ_Variable_1022      snr:0.004156459657178778
        Acuity: Add_Add_74_96_out0_nchw_1_1024_14_14.qnt.tensor                                  ORT-CPU: Add_74->PPQ_Variable_1106      snr:0.0035067689479688683
        Acuity: Add_Add_81_81_out0_nchw_1_1024_14_14.qnt.tensor                                  ORT-CPU: Add_81->PPQ_Variable_1190      snr:0.002817671942489091
        Acuity: Add_Add_88_66_out0_nchw_1_1024_14_14.qnt.tensor                                  ORT-CPU: Add_88->PPQ_Variable_1274      snr:0.0011721558990762662
        Acuity: Add_Add_95_51_out0_nchw_1_1024_14_14.qnt.tensor                                  ORT-CPU: Add_95->PPQ_Variable_1358      snr:0.14955186754757677
        Acuity: Add_Add_9_231_out0_nchw_1_256_56_56.qnt.tensor                                   ORT-CPU: Add_9->PPQ_Variable_338        snr:0.0012610773344295155
        Acuity: Conv_Conv_0_252_out0_nchw_1_64_112_112.qnt.tensor                                ORT-CPU: Conv_0->PPQ_Variable_236       snr:0.08278019995627235
        Acuity: Conv_Conv_101_44_out0_nchw_1_2048_7_7.qnt.tensor                                 ORT-CPU: Conv_101->PPQ_Variable_1412    snr:0.003136722764714228
        Acuity: Conv_Conv_102_45_out0_nchw_1_2048_7_7.qnt.tensor                                 ORT-CPU: Conv_102->PPQ_Variable_1430    snr:0.0017023306991021544
        Acuity: Conv_Conv_105_46_out0_nchw_1_512_7_7.qnt.tensor                                  ORT-CPU: Conv_105->PPQ_Variable_1478    snr:0.1507590433999134
        Acuity: Conv_Conv_107_37_out0_nchw_1_512_7_7.qnt.tensor                                  ORT-CPU: Conv_107->PPQ_Variable_1496    snr:0.09256327832218221
        Acuity: Conv_Conv_109_30_out0_nchw_1_2048_7_7.qnt.tensor                                 ORT-CPU: Conv_109->PPQ_Variable_1514    snr:0.002529668247420068
        Acuity: Conv_Conv_112_29_out0_nchw_1_512_7_7.qnt.tensor                                  ORT-CPU: Conv_112->PPQ_Variable_1562    snr:0.13772802074685114
        Acuity: Conv_Conv_114_21_out0_nchw_1_512_7_7.qnt.tensor                                  ORT-CPU: Conv_114->PPQ_Variable_1580    snr:0.07332842305562158
        Acuity: Conv_Conv_116_15_out0_nchw_1_2048_7_7.qnt.tensor                                 ORT-CPU: Conv_116->PPQ_Variable_1598    snr:0.0017119333134883116
        Acuity: Conv_Conv_11_241_out0_nchw_1_64_56_56.qnt.tensor                                 ORT-CPU: Conv_11->PPQ_Variable_368      snr:0.025791024035917865
        Acuity: Conv_Conv_13_232_out0_nchw_1_64_56_56.qnt.tensor                                 ORT-CPU: Conv_13->PPQ_Variable_386      snr:0.026695672119450425
        Acuity: Conv_Conv_15_224_out0_nchw_1_256_56_56.qnt.tensor                                ORT-CPU: Conv_15->PPQ_Variable_404      snr:0.0005407498126977094
        Acuity: Conv_Conv_18_226_out0_nchw_1_64_56_56.qnt.tensor                                 ORT-CPU: Conv_18->PPQ_Variable_452      snr:0.08467941043753574
        Acuity: Conv_Conv_20_217_out0_nchw_1_64_56_56.qnt.tensor                                 ORT-CPU: Conv_20->PPQ_Variable_470      snr:0.2174488620367163
