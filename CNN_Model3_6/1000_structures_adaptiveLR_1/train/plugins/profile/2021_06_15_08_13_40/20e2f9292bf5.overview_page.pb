?	?S?[$s?@?S?[$s?@!?S?[$s?@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?S?[$s?@A?M?G?@1?ù?҈@I??I?1@rEagerKernelExecute 0*	S㥛H?@2T
Iterator::Prefetch::Generator?E?xD0@!Xh6?m:J@)?E?xD0@1Xh6?m:J@:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2
K<?lB-@!?|+??G@)l??g?!@1M\?@<@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? ׽?	z@!??K???2@)׽?	z@1??K???2@:Preprocessing2I
Iterator::Prefetch*t^c????!b?'?ܷ??)*t^c????1b?'?ܷ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?'?XQ???!?3ґ??)?'?XQ???1?3ґ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?l??p??!??KA??)??fd????15?S????:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard??[1P-@!>??"1?G@)\:?<c_??1??ja"???:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatchv?SG-@!?%2ɚG@)??-?R??1vD5؋??:Preprocessing2F
Iterator::Model&?B????!ȝ??m??)?????r{?1??D<!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??\?޸@Q??	9bX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A?M?G?@A?M?G?@!A?M?G?@      ??!       "	?ù?҈@?ù?҈@!?ù?҈@*      ??!       2      ??!       :	??I?1@??I?1@!??I?1@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??\?޸@y??	9bX@?"?
wgradient_tape/model_6/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????????!????????0"?
wgradient_tape/model_6/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter
l, ^=??!ʧ?y a??0"?
wgradient_tape/model_6/TCN/residual_block_6/weight_normalization_13/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ǞS?`??!????0"?
wgradient_tape/model_6/TCN/residual_block_6/weight_normalization_12/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??J˻?!???f??0"?
wgradient_tape/model_6/TCN/residual_block_5/weight_normalization_11/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterF?L??n??!?oK0m???0"?
wgradient_tape/model_6/TCN/residual_block_5/weight_normalization_10/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?p??6??!?????P??0"?
vgradient_tape/model_6/TCN/residual_block_4/weight_normalization_8/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilterb??lP??!Y,T \#??0"?
vgradient_tape/model_6/TCN/residual_block_4/weight_normalization_9/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!фѮ???0"?
vgradient_tape/model_6/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropInputConv2DBackpropInputSK??Tnw?!h??W???0"?
vgradient_tape/model_6/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropInputConv2DBackpropInput(٢?1aw?!????F??0Q      Y@Y?????AU@a?;?!??-@q?q?Yn???y??m?e?:?"?	
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 