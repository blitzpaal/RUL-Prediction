  *	V=}?@2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV20du??d@!+]-?Q@)?/?rn@1?h	9?C@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? iƢ??	@!???FB?>@)iƢ??	@1???FB?>@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?/ע(??!N??"??2@)?/ע(??1N??"??2@:Preprocessing2T
Iterator::Prefetch::Generator?(??=d??!???o?#@)?(??=d??1???o?#@:Preprocessing2F
Iterator::Model??N??z @!???P-?3@)??O9&???14?7?h??:Preprocessing2I
Iterator::Prefetch????????!?Ǩ????)????????1?Ǩ????:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard??Ir@!?W?%?Q@)S	O??'??1xr?TEq??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???L=??!?????2@)???1ZGu?1??6??Z??:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::RebatchD? ?i@!n??	?Q@)?O?I?5s?10??+????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.