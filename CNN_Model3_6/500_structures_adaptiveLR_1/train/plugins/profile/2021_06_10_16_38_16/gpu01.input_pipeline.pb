  *	8?A`%??@2?
IIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2XU/??$@!1+?q9?T@)O?Z?7Z@1???E@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::BatchV2::TensorSlice? b?o?
@![wT?\?C@)b?o?
@1[wT?\?C@:Preprocessing2T
Iterator::Prefetch::GeneratorFж?5??!?+?)X[0@)Fж?5??1?+?)X[0@:Preprocessing2I
Iterator::Prefetch??i?:??!??#?x???)??i?:??1??#?x???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchT?4??-??!K??J????)T?4??-??1K??J????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?|A	??!;?.횳??)д??h???1TfH?L??:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard	3?z1@!$f????T@)Vb????{?1Vu冉??:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatchp?܁*@!?,_h?T@)$`tys?v?1/?v뵻??:Preprocessing2F
Iterator::Model0??&???!<$+????)???)Wxg?1 ??I??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.