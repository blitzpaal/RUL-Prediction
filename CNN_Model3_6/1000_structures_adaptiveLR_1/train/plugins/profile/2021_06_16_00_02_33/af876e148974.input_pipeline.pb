	/?$???@/?$???@!/?$???@      ??!       "?
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
	h?,;?{@h?,;?{@!h?,;?{@      ??!       "	X SN݈@X SN݈@!X SN݈@*      ??!       2	ݲC?Ö??ݲC?Ö??!ݲC?Ö??:	x?~?~2@x?~?~2@!x?~?~2@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qv?Uy?_B@y?*??W?O@