	|???^?@|???^?@!|???^?@      ??!       "{
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
	?q??{@?q??{@!?q??{@      ??!       "	5??$??@5??$??@!5??$??@*      ??!       2      ??!       :	?u??Ž)@?u??Ž)@!?u??Ž)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?#,?s???yrO?0R?X@