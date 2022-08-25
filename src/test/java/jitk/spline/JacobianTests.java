/*-
 * #%L
 * JavaITK thin plate spline.
 * %%
 * Copyright (C) 2014 - 2022 Howard Hughes Medical Institute.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
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
