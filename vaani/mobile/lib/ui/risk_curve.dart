import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class RiskCurve extends StatelessWidget {
  const RiskCurve({super.key, required this.emaHistory, required this.threshold});
  final List<double> emaHistory;
  final double threshold;

  @override
  Widget build(BuildContext context) {
    if (emaHistory.isEmpty) {
      return const SizedBox(height: 120, child: Center(child: Text('No data yet')));
    }
    final spots = [
      for (var i = 0; i < emaHistory.length; i++) FlSpot(i.toDouble(), emaHistory[i])
    ];
    return SizedBox(
      height: 160,
      child: LineChart(
        LineChartData(
          minY: 0,
          maxY: 1,
          extraLinesData: ExtraLinesData(horizontalLines: [
            HorizontalLine(y: threshold, color: Colors.amber, strokeWidth: 1),
          ]),
          lineBarsData: [
            LineChartBarData(spots: spots, isCurved: false, dotData: const FlDotData(show: false)),
          ],
        ),
      ),
    );
  }
}
