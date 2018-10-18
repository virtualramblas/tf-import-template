package org.googlielmo.tfimport;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/**
 * Imports a pre-trained TensorFlow model and makes predictions.
 * 
 * @author Guglielmo Iozzia
 *
 */
public class TensorFlowModelImporter {

	/** The filename of the TensorFlow serialized model to import. */
	private String modelFileName;
	/** Entry point for ND4J autodiff. */
	private SameDiff sameDiff;
	
	/** Default constructor. **/
	public TensorFlowModelImporter() {
		this.modelFileName = "";
	}
	
	/**
	 * Constructor.
	 * @param modelFileName The filename of the TensorFlow serialized model to import.
	 */
	public TensorFlowModelImporter(String modelFileName) {
		this.modelFileName = modelFileName;
	}

	/**
	 * Imports a TensorFlow serialized model to make it ready to be used on the JVM. 
	 * @throws Exception
	 */
	public void loadModel() throws Exception {
		File file = new ClassPathResource(modelFileName).getFile();
		sameDiff = TFGraphMapper.getInstance().importGraph(file);
	}
	
	/**
	 * Imports a TensorFlow serialized model to make it ready to be used on the JVM. 
	 * @param modelFileName The filename of the TensorFlow serialized model to import.
	 * @throws Exception
	 */
	public void loadModel(String modelFileName) throws Exception {
		setModelFileName(modelFileName);
		loadModel();
	}
	
	/**
	 * Does predictions using the imported TensorFlow model. 
	 * @param array Input INDArray
	 * @return INDArray
	 */
	public INDArray predict(INDArray array) {
		INDArray batchedArray = Nd4j.expandDims(array, 0);
		sameDiff.associateArrayWithVariable(batchedArray, sameDiff.variables().get(0));
        INDArray outArray = sameDiff.execAndEndResult();
        return outArray;
	}
	
	/** Getter for modelFileName. */
	public String getModelFileName() {
		return modelFileName;
	}

	/** 
	 * Setter for modelFileName
	 * @param modelFileName
	 */
	public void setModelFileName(String modelFileName) {
		this.modelFileName = modelFileName;
	}

	/** Getter for sameDiff. */
	public SameDiff getSd() {
		return sameDiff;
	}

}
