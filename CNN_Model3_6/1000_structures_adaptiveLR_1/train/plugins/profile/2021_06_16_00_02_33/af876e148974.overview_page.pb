?	/?$???@/?$???@!/?$???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC/?$???@h?,;?{@1X SN݈@AݲC?Ö??Ix?~?~2@rEagerKernelExecute 0*	??ʡ@?@2T
Iterator::Prefetch::Generator??E
?%@!v?ۆzE@)??E
?%@1v?ۆzE@:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2D?o?-@!qཌྷ8?L@)B?Ѫ?D@1~???SC=@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? GZ*oG?@!f??&?;@)GZ*oG?@1f??&?;@:Preprocessing2I
Iterator::Prefetch?]/M???!
NҤ???)?]/M???1
NҤ???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA?S?????!^????˹?)A?S?????1^????˹?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????6???!+??SR ??)W?sD?K??1?<	?t??:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard?-?R?-@!'?As?L@)9?#+???1??s?UQ??:Preprocessing2F
Iterator::Model??t 멱?!???R???)??>???1?????:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch??mnL?-@!???ʛL@)1A?º??1-*?$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIv?Uy?_B@Q?*??W?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	h?,;?{@h?,;?{@!h?,;?{@      ??!       "	X SN݈@X SN݈@!X SN݈@*      ??!       2	ݲC?Ö??ݲC?Ö??!ݲC?Ö??:	x?~?~2@x?~?~2@!x?~?~2@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qv?Uy?_B@y?*??W?O@?"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilterJ???0M??!J???0M??0"?
ugradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??g?1??! ^?xc???0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_12/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???䮼?!??3@???0"?
ugradient_tape/model/TCN/residual_block_6/weight_normalization_13/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??.????!U?Vi??0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_11/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter<t2u?9??!?<Z?????0"?
ugradient_tape/model/TCN/residual_block_5/weight_normalization_10/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????ɯ??!?{???g??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_9/compute_weights/conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Z%??*??!????DA??0"?
tgradient_tape/model/TCN/residual_block_4/weight_normalization_8/compute_weights/conv1D_0/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Zp[?S??!a)?????0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_15/compute_weights/conv1D_1/conv1d/Conv2DBackpropInputConv2DBackpropInput?Kw8^w?!??????0"?
tgradient_tape/model/TCN/residual_block_7/weight_normalization_14/compute_weights/conv1D_0/conv1d/Conv2DBackpropInputConv2DBackpropInput?V9??Ww?!??2<PI??0Q      Y@Yg?(?@U@a?d??V?-@qB??g?K@y???!?;?"?

both?Your program is POTENTIALLY input-bound because 35.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?55.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 