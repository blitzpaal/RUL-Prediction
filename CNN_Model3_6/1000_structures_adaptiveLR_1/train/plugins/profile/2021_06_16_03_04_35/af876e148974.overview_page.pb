?	\?	?_?@\?	?_?@!\?	?_?@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:\?	?_?@w?n??@1ԶaԈ@IF?2?X.@rEagerKernelExecute 0*~j?t???@)      ?=2T
Iterator::Prefetch::Generator?\S ??#@!|?'?JE@)?\S ??#@1|?'?JE@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? p??R?]@!?E?ۤg<@)p??R?]@1?E?ۤg<@:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2?7?V*@!?:??,`L@)?uqP@190?;?X<@:Preprocessing2I
Iterator::Prefetchj?TQ?ʪ?!`sw????)j?TQ?ʪ?1`sw????:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch?%?h*@!?>w?rL@)E?????1??G?]??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchs+??X?!q???Ӽ?)s+??X?1q???Ӽ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?9z?ަ?!??Zۉ???)???????1ߓ?*>s??:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard?Բ??p*@!r????{L@)?im?k??1?^?ڒĲ?:Preprocessing2F
Iterator::Model?m?2d??!/?(On??))\???(|?1:\?e*V??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI B*5Z.@Q??V.?vX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?n??@w?n??@!w?n??@      ??!       "	ԶaԈ@ԶaԈ@!ԶaԈ@*      ??!       2      ??!       :	F?2?X.@F?2?X.@!F?2?X.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q B*5Z.@y??V.?vX@?"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterZ????I??!Z????I??0"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter#????4??!>?v?A???0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_12/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilterT??|???!???t????0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_13/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???Xw??!?
	 re??0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_10/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?b/?p??!28????0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_11/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??i?y??!?	?!d??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_9/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd??r???!?	ð?A??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_8/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:??!???z5???0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropInputConv2DBackpropInputD~?N?pw?!??51??0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropInputConv2DBackpropInput.7??Qw?!#????G??0Q      Y@Yg?(?@U@a?d??V?-@q?#Y?bN@y?????e:?"?

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
Refer to the TF2 Profiler FAQb?60.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 