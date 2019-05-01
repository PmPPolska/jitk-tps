package jitk.spline;

import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class SpeedMemoryOptimization {

	double[][] srcPts;
	double[][] tgtPts;

	int ndims;
	int N;     // # landmarks
	double ptScale = 10;
	double offScale = 1; // scale for offsets
	
	public static Random rand = new Random( 31415926536l );
	public Logger logger = LogManager.getLogger(SpeedMemoryOptimization.class.getName());
	
	public SpeedMemoryOptimization(int N, int ndims){
		setup(N,ndims,ptScale);
	}
	
	public SpeedMemoryOptimization(int N, int ndims, double scale){
		setup(N,ndims,scale);
	}
	
	/**
	 * 
	 */
	public void setup(int N, int ndims, double scale){
		logger.info("setup double");
		this.N = N;
		this.ndims = ndims;
		this.ptScale = scale;
		srcPts = new double[ndims][N];
		tgtPts = new double[ndims][N];
		
		for (int d=0; d<ndims; d++) for( int i=0; i<N; i++ )
		{
			srcPts[d][i] = scale * rand.nextDouble();
			tgtPts[d][i] = srcPts[d][i] + offScale * rand.nextDouble();
		}
	}
	
	/**
	 * Tried varying the linear system solver
	 * (in KernelTransform) for speed
	 */
	public void speed() {

		long startTime = System.currentTimeMillis();

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);

		long endTime = System.currentTimeMillis();
		logger.info("sep (N=" + N + ") total time: " + (endTime - startTime) + "ms");

		double[] pt = new double[ndims];
		double[] result = new double[ndims];

		for (int i = 0; i < N; i++) {
			for (int d = 0; d < ndims; d++) {
				pt[d] = srcPts[d][i];
			}
			tps.apply(pt, result);
		}
	}

	public long testKernelSpeedupApply( final int N, final double scale, final Random random )
	{
		long startTime = System.currentTimeMillis();
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		
		long endTime = System.currentTimeMillis();
		//System.out.println("sep (N="+N+") total time: " + (endTime-startTime) + "ms" );
		
		double[] pt = new double[ndims];
		double[] result = new double[ndims];
		
		startTime = System.currentTimeMillis();
		for ( int i=0; i<N; i++)
		{
			int j = random.nextInt( pt.length );

			for (int d=0; d<ndims; d++){
				pt[d]  = srcPts[d][j] + scale * random.nextDouble();
			}
			tps.apply( pt, result );
		}

		endTime = System.currentTimeMillis();
		
		return endTime-startTime;

	}

	public static void testKernelSpeedup()
	{
		double scale = 10;

		int N = 1000000;
		int Niters = 20;
		double[] displacement = new double[ 3 ];

		
		Random rand = new Random( 0l );
		for( int j = 0; j < Niters; j++ )
		{
			long startTimeNew = System.currentTimeMillis();

			for( int i = 0; i < N; i++ )
			{
				for( int d = 0; d < 3; d++ )
				{
					displacement[ d ] = scale * rand.nextDouble();
				}

				ThinPlateR2LogRSplineKernelTransform.r2LogrFromDisplacement( displacement );
			}

			long endTimeNew = System.currentTimeMillis();
			long runTimeNew = endTimeNew - startTimeNew;
			System.out.println( " " + runTimeNew );
		}

		System.out.println( " " );
		rand = new Random( 0l );

		for( int j = 0; j < Niters; j++ )
		{
			long startTime = System.currentTimeMillis();
			for( int i = 0; i < N; i++ )
			{
				for( int d = 0; d < 3; d++ )
				{
					displacement[ d ] = scale * rand.nextDouble();
				}

				double r = ThinPlateR2LogRSplineKernelTransform.normSqrd( displacement );
				ThinPlateR2LogRSplineKernelTransform.r2Logr( Math.sqrt( r ));
			}
			long endTime = System.currentTimeMillis();
			long runTimeOld = endTime - startTime;
		}
	}
	
	public static void main(String[] args) {
		
		System.out.println("starting");
		
		int[] NList = new int[]{100, 200, 400, 800, 1200 };
		
		int ndims = 3;
		int numTrials = 35;
		
		int NtestPoints = 10000;
		double scale = 5.0;
		
		int numToSkip = 5;

		Random random = new Random();
		for (int i = 0; i < NList.length; i++) {

			System.out.println("tests with " + NList[i] + " landmarks");
			SpeedMemoryOptimization smo = new SpeedMemoryOptimization(NList[i], ndims, 10f);
			double avgTime = 0;
			for (int t = 0; t < numTrials; t++) {
				// smo.speed();

				long runtime = smo.testKernelSpeedupApply(NtestPoints, scale, random);

				if (t >= numToSkip) {
					avgTime += runtime;
				}
			}
			avgTime /= (numTrials - numToSkip);
			System.out.println("" + avgTime);
		}

		System.out.println("finished");
		System.exit(0);
	}

}
