package jitk.spline;

import static org.junit.Assert.assertEquals;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.junit.Test;

public class JacobianTests
{

	double[][] srcPts;

	double[][] tgtPts;

	double trueDet;

	int ndims;

	int N;

	public void genPtListSimple2d()
	{
		ndims = 2;
		srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0 } };	// y

		tgtPts = new double[][] {
				{ -2.0, 0.0, 2.0, 0.0 },	// x
				{ 0.0, -2.0, 0.0, 2.0 } };	// y
	}

	@Test
	public void testJacobianAffine()
	{
		genPtListSimple2d();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );

		double[] p = new double[ 2 ];
		double[] q = new double[ 2 ];
		int i = 0;
		for ( double x = -4; x < 4; x++ )
			for ( double y = -4; y < 4; y++ )
			{
				p[ 0 ] = x;
				p[ 1 ] = y;

				tps.apply( p, q );
				double[][] j = tps.jacobian( p );
				DenseMatrix64F jacobianMtx = new DenseMatrix64F( j );
				double jdet = CommonOps.det( jacobianMtx );

				assertEquals( "Affine determinant pt (" + i + ")", 4, jdet, 0.01 );
				i++;
			}
	}
}
