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

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.NormOps_DDRM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TransformInverseGradientDescent
{
	int ndims;

	ThinPlateR2LogRSplineKernelTransform xfm;

	DMatrixRMaj jacobian;

	DMatrixRMaj directionalDeriv; // derivative in direction of dir (the
										// descent direction )

	DMatrixRMaj descentDirectionMag; // computes dir^T directionalDeriv
										// (where dir^T is often
										// -directionalDeriv)

	DMatrixRMaj dir; // descent direction

	DMatrixRMaj errorV; // error vector ( errorV = target - estimateXfm )

	DMatrixRMaj estimate; // current estimate

	DMatrixRMaj estimateXfm; // current estimateXfm

	DMatrixRMaj target;

	double error = 9999.0;

	double stepSz = 1.0;

	int maxIters = 20;

	double eps = 1e-6;

	double beta = 0.7;

	protected static Logger logger = LoggerFactory.getLogger(
			TransformInverseGradientDescent.class );

	public TransformInverseGradientDescent( int ndims, ThinPlateR2LogRSplineKernelTransform xfm )
	{
		this.ndims = ndims;
		this.xfm = xfm;
		dir = new DMatrixRMaj( ndims, 1 );
		errorV = new DMatrixRMaj( ndims, 1 );
		directionalDeriv = new DMatrixRMaj( ndims, 1 );
		descentDirectionMag = new DMatrixRMaj( 1, 1 );
	}

	public void setEps( double eps )
	{
		this.eps = eps;
	}

	public void setStepSize( double stepSize )
	{
		stepSz = stepSize;
	}

	public void setJacobian( double[][] mtx )
	{
		this.jacobian = new DMatrixRMaj( mtx );
		logger.trace( "setJacobian:\n" + this.jacobian );
	}

	public void setTarget( double[] tgt )
	{
		this.target = new DMatrixRMaj( ndims, 1 );
		target.setData( tgt );
	}

	public DMatrixRMaj getErrorVector()
	{
		return errorV;
	}

	public DMatrixRMaj getDirection()
	{
		return dir;
	}

	public DMatrixRMaj getJacobian()
	{
		return jacobian;
	}

	public void setEstimate( double[] est )
	{
		this.estimate = new DMatrixRMaj( ndims, 1 );
		estimate.setData( est );
	}

	public void setEstimateXfm( double[] est )
	{
		this.estimateXfm = new DMatrixRMaj( ndims, 1 );
		estimateXfm.setData( est );
		updateError();
	}

	public DMatrixRMaj getEstimate()
	{
		return estimate;
	}

	public double getError()
	{
		return error;
	}

	public void oneIteration()
	{
		oneIteration( true );
	}

	public void oneIteration( boolean updateError )
	{
		// at this point, we need a target, an estimate, and a derivative matrix
		computeDirection();
		updateEstimate( stepSz );
		if ( updateError )
			updateError();
	}

	/**
	 * Computes 2A^T(Ax - b ) using the current matrix as A, the current error
	 * vector as b, and the current estimate as x
	 */
	public void computeDirectionSteepest()
	{
		final DMatrixRMaj tmp = new DMatrixRMaj( ndims, 1 );

		logger.trace( "\nerrorV:\n" + errorV );

		CommonOps_DDRM.mult( jacobian, estimate, tmp );
		// TODO this line is wrong isnt it
		CommonOps_DDRM.subtractEquals( tmp, errorV );

		// now tmp contains Ax-b
		CommonOps_DDRM.multTransA( 2, jacobian, tmp, dir );

		// normalize dir
		final double norm = NormOps_DDRM.normP2( dir );
		// normalize
		// TODO put in a check if norm is too small
		CommonOps_DDRM.divide( norm, dir );

		// compute the directional derivative
		CommonOps_DDRM.mult( jacobian, dir, directionalDeriv );

		// go in the negative gradient direction to minimize cost
		CommonOps_DDRM.scale( -1, dir );
	}

	public void computeDirection()
	{
		CommonOps_DDRM.solve( jacobian, errorV, dir );

		final double norm = NormOps_DDRM.normP2( dir );
		CommonOps_DDRM.divide( norm, dir );

		// compute the directional derivative
		CommonOps_DDRM.mult( jacobian, dir, directionalDeriv );

		//
		CommonOps_DDRM.multTransA( dir, directionalDeriv, descentDirectionMag );

		logger.debug( "descentDirectionMag: " + descentDirectionMag.get( 0 ) );
	}

	/**
	 * Uses Backtracking Line search to determine a step size.
	 *
	 * @param c the armijoCondition parameter
	 * @param beta the fraction to multiply the step size at each iteration ( less than 1 )
	 * @param maxtries max number of tries
	 * @param t0 initial step size
	 * @return the step size
	 */
	public double backtrackingLineSearch( double c, double beta, int maxtries, double t0 )
	{
		double t = t0; // step size

		int k = 0;
		// boolean success = false;
		while ( k < maxtries )
		{
			if ( armijoCondition( c, t ) )
			{
				// success = true;
				break;
			}
			else
				t *= beta;

			k++;
		}

		logger.trace( "selected step size after " + k + " tries" );

		return t;
	}

	/**
	 * Returns true if the armijo condition is satisfied.
	 *
	 * @param c the c parameter
	 * @param t the step size
	 * @return true if the step size satisfies the condition
	 */
	public boolean armijoCondition( double c, double t )
	{
		final double[] d = dir.data;
		final double[] x = estimate.data; // give a convenient name

		final double[] x_ap = new double[ ndims ];
		for ( int i = 0; i < ndims; i++ )
			x_ap[ i ] = x[ i ] + t * d[ i ];

		// don't have to do this in here - this should be reused
		// double[] phix = xfm.apply( x );
		// TODO make sure estimateXfm is updated at the correct time
		final double[] phix = estimateXfm.data;
		final double[] phix_ap = xfm.apply( x_ap );

		final double fx = squaredError( phix );
		final double fx_ap = squaredError( phix_ap );

		// descentDirectionMag is a scalar
		// computeExpectedDescentReduction();
//		CommonOps_DDRM.multTransA( dir, directionalDeriv, descentDirectionMag );
//		logger.debug( "descentDirectionMag: " + descentDirectionMag.get( 0 ) );

		final double m = sumSquaredErrorsDeriv( this.target.data, phix ) * descentDirectionMag.get( 0 );

		logger.trace( "   f( x )     : " + fx );
		logger.trace( "   f( x + ap ): " + fx_ap );
//		logger.debug( "   p^T d      : " + descentDirectionMag.get( 0 ));
//		logger.debug( "   m          : " + m );
//		logger.debug( "   c * m * t  : " + c * t * m );
		logger.trace( "   f( x ) + c * m * t: " + ( fx + c * t * m ) );

		if ( fx_ap < fx + c * t * m )
			return true;
		else
			return false;
	}

	public double squaredError( double[] x )
	{
		double error = 0;
		for ( int i = 0; i < ndims; i++ )
			error += ( x[ i ] - this.target.get( i ) ) * ( x[ i ] - this.target.get( i ) );

		return error;
	}

	public void updateEstimate( double stepSize )
	{
		logger.trace( "step size: " + stepSize );
		logger.trace( "estimate:\n" + estimate );

		// go in the negative gradient direction to minimize cost
//		CommonOps_DDRM.scale( -stepSize / norm, dir );
//		CommonOps_DDRM.addEquals( estimate, dir );

		// dir should be pointing in the descent direction
		CommonOps_DDRM.addEquals( estimate, stepSize, dir );

		logger.trace( "new estimate:\n" + estimate );
	}

	public void updateEstimateNormBased( double stepSize )
	{
		logger.debug( "step size: " + stepSize );
		logger.trace( "estimate:\n" + estimate );

		final double norm = NormOps_DDRM.normP2( dir );
		logger.debug( "norm: " + norm );

		// go in the negative gradient direction to minimize cost
		if ( norm > stepSize )
		{
			CommonOps_DDRM.scale( -stepSize / norm, dir );
		}

		CommonOps_DDRM.addEquals( estimate, dir );

		logger.trace( "new estimate:\n" + estimate );
	}

	public void updateError()
	{
		if ( estimate == null || target == null )
		{
			System.err.println( "WARNING: Call to updateError with null target or estimate" );
			return;
		}

		// errorV = estimate - target
//		CommonOps_DDRM.sub( estimateXfm, target, errorV );

		// ( errorV = target - estimateXfm  )
		CommonOps_DDRM.subtract( target, estimateXfm, errorV );

		logger.trace( "#########################" );
		logger.trace( "updateError, estimate   :\n" + estimate );
		logger.trace( "updateError, estimateXfm:\n" + estimateXfm );
		logger.trace( "updateError, target     :\n" + target );
		logger.trace( "updateError, error      :\n" + errorV );
		logger.trace( "#########################" );

		// set scalar error equal to max of component-wise errors
		error = Math.abs( errorV.get( 0 ) );
		for ( int i = 1; i < ndims; i++ )
		{
			if ( Math.abs( errorV.get( i ) ) > error )
				error = Math.abs( errorV.get( i ) );
		}

	}

	/**
	 * This function returns \nabla f ^T \nabla f where f = || y - x ||^2 and
	 * the gradient is taken with respect to x
	 *
	 * @param y
	 * @param x
	 * @return
	 */
	private double sumSquaredErrorsDeriv( double[] y, double[] x )
	{
		double errDeriv = 0.0;
		for ( int i = 0; i < ndims; i++ )
			errDeriv += ( y[ i ] - x[ i ] ) * ( y[ i ] - x[ i ] );

		return 2 * errDeriv;
	}

	public static double sumSquaredErrors( double[] y, double[] x )
	{
		final int ndims = y.length;

		double err = 0.0;
		for ( int i = 0; i < ndims; i++ )
			err += ( y[ i ] - x[ i ] ) * ( y[ i ] - x[ i ] );

		return err;
	}

	public static void copyVectorIntoArray(final  DMatrixRMaj vec, final double[] array )
	{
		System.arraycopy( vec.data, 0, array, 0, vec.getNumElements() );
	}

}
