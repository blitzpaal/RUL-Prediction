	\?	?_?@\?	?_?@!\?	?_?@      ??!       "{
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
	w?n??@w?n??@!w?n??@      ??!       "	ԶaԈ@ԶaԈ@!ԶaԈ@*      ??!       2      ??!       :	F?2?X.@F?2?X.@!F?2?X.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q B*5Z.@y??V.?vX@