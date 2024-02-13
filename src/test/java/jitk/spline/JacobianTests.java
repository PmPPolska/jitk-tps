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
import static org.junit.Assert.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;

import java.util.function.Predicate;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.core.IsEqual;
import org.junit.Test;

public class JacobianTests
{
//	public static class PointPairs {
//		int ndims;
//		double[][] srcPts;
//		double[][] tgtPts;
//
//		public PointPairs( int ndims, double[][] src, double[][] tgt )
//		{
//			this.ndims = ndims;
//			srcPts = src;
//			tgtPts = tgt;
//		}
//	}

	public ThinPlateR2LogRSplineKernelTransform expandingBy4Dim2() {

		final int ndims = 2;
		final double[][] srcPts = new double[][]{
				{-1.0, 0.0, 1.0, 0.0}, // x
				{0.0, -1.0, 0.0, 1.0}}; // y

		final double[][] tgtPts = new double[][]{
				{-2.0, 0.0, 2.0, 0.0}, // x
				{0.0, -2.0, 0.0, 2.0}}; // y

		return new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);
	}

	public ThinPlateR2LogRSplineKernelTransform contractingBy4Dim2()
	{
		final int ndims = 2;
		final double[][] srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0 } };	// y

		final double[][] tgtPts = new double[][] {
				{ -0.5, 0.0, 0.5, 0.0 },	// x
				{ 0.0, -0.5, 0.0, 0.5 } };	// y

		return new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);
	}

	public ThinPlateR2LogRSplineKernelTransform genPtListIdentity2d()
	{
		final int ndims = 2;
		final double[][] srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0, 0.0 } };	// y

		final double[][] tgtPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.0 },	// x
				{ 0.0, -1.0, 0.0, 1.0, 0.0 } };	// y

		return new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);
	}

	public ThinPlateR2LogRSplineKernelTransform genPtListNonlinearSmall()
	{
		final int ndims = 2;
		final double[][] srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0, 0.0 } };	// y

		final double[][] tgtPts = new double[][] {
				{ -1.1, 0.0, 1.1, 0.0,  0.1 },	// x
				{ 0.0, -1.1, 0.0, 1.1, -0.1 } };	// y

		return new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);
	}

	public ThinPlateR2LogRSplineKernelTransform genPtListNonlinear2d()
	{
		final int ndims = 2;
		final double[][] srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0, 0.0 } };	// y

		final double[][] tgtPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.5 },	// x
				{ 0.0, -1.0, 0.0, 1.0, 0.5 } };	// y

		return new ThinPlateR2LogRSplineKernelTransform(ndims, srcPts, tgtPts);
	}

	private void helper2d(double min, double max, Matcher<Double> matcher, ThinPlateR2LogRSplineKernelTransform tps)
	{
		System.out.println( "helper 2d");
		final double[] p = new double[ 2 ];
		int i = 0;
		for ( double x = min; x <= max; x++ )
			for ( double y = min; y <= max; y++ )
			{
				p[ 0 ] = x;
				p[ 1 ] = y;
				final double[][] j = tps.jacobian( p );
				final DMatrixRMaj jacobianMtx = new DMatrixRMaj( j );
				final double jdet = CommonOps_DDRM.det( jacobianMtx );

				final String s = String.format("jacobian determinant at (%.2f, %.2f) = %,4f ", x, y, jdet );
				System.out.println( s );
				System.out.println( jacobianMtx );

				System.out.println( "" );
				System.out.println( "" );
				System.out.println( "" );

				assertThat( "jacobian determinant pt (" + i + ")", jdet, matcher );
				i++;
			}
	}

	@Test
	public void testJacobianAffine()
	{
		final double eps = 1e-4;
		helper2d(-4, 4, new EqualsDelta(1,eps), genPtListIdentity2d());
//		helper2d(-4, 4, new EqualsDelta(4,eps), expandingBy4Dim2());
//		helper2d(-4, 4, new EqualsDelta(0.25,eps), contractingBy4Dim2());
	}

	@Test
	public void testJacobianNonlinear()
	{
		final Predicate<Double> gt0 = x -> x > 0.0;
//		helper2d(-4, 4, new PredicateMatcher( gt0 ), genPtListNonlinearSmall());
		helper2d(-4, 4, new PredicateMatcher( gt0 ), genPtListNonlinear2d());
//		helper2d(-4, 4, x -> x > 0.0, genPtListNonlinear2d());
	}

	public class EqualsDelta extends PredicateMatcher {

		Predicate<Double> predicate;

		public EqualsDelta( final double value, final double eps )
		{
			super( x -> { return Math.abs( x - value ) < eps; });
		}
	}

	public class PredicateMatcher implements Matcher<Double> {

		Predicate<Double> predicate;

		public PredicateMatcher( Predicate<Double> predicate )
		{
			this.predicate = predicate;
		}

		@Override
		public void describeTo(Description description) {

		}

		@Override
		public boolean matches(Object item) {

			if( item instanceof Double )
				return predicate.test((Double)item);

			return false;
		}

		@Override
		public void describeMismatch(Object item, Description mismatchDescription) {

			// TODO Auto-generated method stub

		}

		@Override
		public void _dont_implement_Matcher___instead_extend_BaseMatcher_() {

			// TODO Auto-generated method stub

		}

	}
}
