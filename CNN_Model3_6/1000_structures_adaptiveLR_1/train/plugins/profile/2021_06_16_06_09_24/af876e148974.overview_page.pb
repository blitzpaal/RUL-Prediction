?	|???^?@|???^?@!|???^?@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:|???^?@?q??{@15??$??@I?u??Ž)@rEagerKernelExecute 0*	Vu?@2T
Iterator::Prefetch::Generator&??'D@!?q0??A@)&??'D@1?q0??A@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? ?R	?@!
??ٻ@@)?R	?@1
??ٻ@@:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2?aot'@!?V?f?O@)	4??)@1?ό?Ua?@:Preprocessing2I
Iterator::Prefetch3NCT?ϰ?!?;????)3NCT?ϰ?1?;????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?[Ɏ?@??!?	g?#??)?[Ɏ?@??1?	g?#??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism_?????!%	N?? ??)?H?]ۛ?1S5?????:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard6sHj??'@!? ??W?O@)??]gE??1???y@v??:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::RebatchND??~z'@!5%????O@)???v?>??1?txl??:Preprocessing2F
Iterator::Model?3????!b?ݱ??)k??????1?)??g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?#,?s???QrO?0R?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q??{@?q??{@!?q??{@      ??!       "	5??$??@5??$??@!5??$??@*      ??!       2      ??!       :	?u??Ž)@?u??Ž)@!?u??Ž)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?#,?s???yrO?0R?X@?"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFiltergbL?R??!gbL?R??0"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter9???!P(gb?-??0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_12/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?5zV?ļ?! o6m_???0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_13/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?y?ػ??!4ׇ?]??0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_11/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ʷ邧?!ݚ	???0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_10/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?S?mm??! t??L??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_9/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?s?????!?k?)??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_8/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter!??a????!C?'?????0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropInputConv2DBackpropInput?@?rXw?!??M}Y??0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropInputConv2DBackpropInput?̃??;w?!_????J??0Q      Y@Yg?(?@U@a?d??V?-@q?[l*?P@y?k!?:?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?67.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 