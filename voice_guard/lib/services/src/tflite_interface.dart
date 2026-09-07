abstract class TFLiteServiceBase {
  bool get isReady;
  Future<void> init();
  double infer(List<double> lfcc, List<double> prosody);
  double scoreChunk(List<double> pcm);
  void dispose();
}
