	?S?[$s?@?S?[$s?@!?S?[$s?@      ??!       "{
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
	A?M?G?@A?M?G?@!A?M?G?@      ??!       "	?ù?҈@?ù?҈@!?ù?҈@*      ??!       2      ??!       :	??I?1@??I?1@!??I?1@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??\?޸@y??	9bX@