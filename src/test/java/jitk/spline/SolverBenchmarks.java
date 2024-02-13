package jitk.spline;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;


@State( Scope.Thread )
@Fork( 1 )
public class SolverBenchmarks {

	@Param({"linear", "SVD"})
	public String solverType;

	@Param({"5", "14"})
	public int landmarksPerDim;

	public double[][] mvgLandmarks;
	public double[][] tgtLandmarks;
	public LinearSolverDense< DMatrixRMaj > solver;

	@Setup
	public void setup()
	{
		final int ndims = 2;
		final int numLandmarks = landmarksPerDim * landmarksPerDim;
		mvgLandmarks = new double[ndims][numLandmarks];
		tgtLandmarks = new double[ndims][numLandmarks];

		int k = 0;
		for( int i = 0; i < landmarksPerDim; i++ ) {
			for( int j = 0; j < landmarksPerDim; j++ ) {

				mvgLandmarks[0][k] = i;
				mvgLandmarks[1][k] = j;

				tgtLandmarks[0][k] = i + (Math.random() - 0.5) * 0.25;
				tgtLandmarks[1][k] = j + (Math.random() - 0.5) * 0.25;

				k++;
			}
		}

		final int nCols = ndims * ( numLandmarks + ndims + 1 );

		if( solverType.equals("SVD"))
			solver = LinearSolverFactory_DDRM.pseudoInverse(false);
		else if( solverType.equals("linear"))
			solver = LinearSolverFactory_DDRM.linear(nCols);
		else
			solver = LinearSolverFactory_DDRM.linear(nCols);
	}


	@Benchmark
	@BenchmarkMode( Mode.AverageTime )
	@OutputTimeUnit( TimeUnit.MILLISECONDS )
	public void bench()
	{
		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( 2, mvgLandmarks, tgtLandmarks, true,
				solver );
	}

	public static void main( final String... args ) throws RunnerException, IOException
	{
		final Options opt = new OptionsBuilder()
				.include( SolverBenchmarks.class.getSimpleName() )
				.warmupIterations( 4 )
				.measurementIterations( 8 )
				.warmupTime( TimeValue.milliseconds( 500 ) )
				.measurementTime( TimeValue.milliseconds( 500 ) )
				.build();
		new Runner( opt ).run();

	}
}
