abstract class TFLiteServiceBase {
  bool get isReady;
  Future<void> init();
  Future<double> infer(List<double> lfcc, List<double> prosody, List<double> physio);
  Future<double> scoreChunk(List<double> pcm);
  void dispose();
}
