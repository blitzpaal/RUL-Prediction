?	????c?@????c?@!????c?@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????c?@k?) Ƴ@1V??D7??@Iq?{??C4@rEagerKernelExecute 0*	z?&1???@2T
Iterator::Prefetch::Generator(v?U{0@!??u?_pD@)(v?U{0@1??u?_pD@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? i?^`V`)@!8??x?@)i?^`V`)@18??x?@:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2?[!?ƪ7@!=???tYM@)????6?%@1a?>y?:;@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????b(??!Y?ȉ·??)???P????1Řȇv??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchB??=Т?!)+ ?T??)B??=Т?1)+ ?T??:Preprocessing2I
Iterator::Prefetcho?!?aK??!??/?og??)o?!?aK??1??/?og??:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard??zݲ7@!O?܈|cM@)^I?\ߗ?1,?????:Preprocessing2F
Iterator::ModelA?M?GŻ?!D}??8??)>?Ӟ?s??1?0?E????:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch?g]???7@!"??\M@)H?`?????17??
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?x6?"@Q2?L??NX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k?) Ƴ@k?) Ƴ@!k?) Ƴ@      ??!       "	V??D7??@V??D7??@!V??D7??@*      ??!       2      ??!       :	q?{??C4@q?{??C4@!q?{??C4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?x6?"@y2?L??NX@?"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?h?TS??!?h?TS??0"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter׷^?4??!Y?zD??0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_12/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??Mp???!O?[{`???0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_13/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?im?Q@??!?ri?j]??0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_10/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??MaH???!?O~8????0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_11/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter*+?M???!HRZ??O??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_9/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterk??k??!Ъ͡O#??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_8/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter֑K섶??!_0????0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropInputConv2DBackpropInputy??F2?w?!???-6??0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropInputConv2DBackpropInput?????w?!* Ԭ9???0Q      Y@Yg?(?@U@a?d??V?-@q?ȶ????y&???:?"?	
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