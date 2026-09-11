abstract class TFLiteServiceBase {
  bool get isReady;
  Future<void> init();
  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars);
  Future<(double, String?, double)> scoreChunk(List<double> pcm);
  void dispose();
}
