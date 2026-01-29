from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import time
import os
import argparse

# configuration variables
models_folder="models"
window_size=24

def main(target_model,output,output_file):
    # reading test dataset
    print("Reading test dataset:",end=" ")
    test_df=pd.read_csv("test-dataset.csv",index_col=0)
    test_df["time"]=test_df["time"].apply(lambda t:datetime.fromisoformat(t))
    print("Ok")

    # loading scaler
    print("Loading scaler:",end=" ")
    scaler=pickle.load(open(os.path.join(models_folder,"scaler.pkl"),"rb"))
    print("Ok")

    # scaling data
    print("Scaling test dataset:",end=" ")
    scaled_test_data= scaler.transform(test_df[["requests"]].values)
    print("Ok")

    results=[]
    if(target_model=="cnn1d_cpu" or target_model=="all"):
        print("Sleeping 10 seconds before starting the benchmark:",end=" ")
        time.sleep(10)
        print("Ok")
        
        # loading lstm model
        print("Loading CNN1D model:",end=" ")        
        lstm_model=load_model(os.path.join(models_folder,"cnn1d.keras"))
        print("Ok")
        
        print("Starting CNN1D on CPU benchmark")
        for i in range(window_size,len(scaled_test_data)):
            # preprocessing
            cnn_input=scaled_test_data[i-window_size:i]
            x=cnn_input.reshape(1,window_size,1)
            
            # performing inference
            print(f"CNN1D-cpu inference step {i}:")
            start_time=time.time_ns()
            prediction=lstm_model.predict(x)
            end_time=time.time_ns()
            print("Ok")

            # postprocessing
            inference_latency=end_time-start_time
            
            # updating results
            if(output): 
                results.append(["cnn1d_cpu",inference_latency,prediction.flatten().tolist()[0],scaled_test_data.flatten().tolist()[i]])

    if(target_model=="lstm_cpu" or target_model=="all"):
        print("Sleeping 10 seconds before starting the benchmark:",end=" ")
        time.sleep(10)
        print("Ok")
        
        # loading lstm model
        print("Loading LSTM model:",end=" ")        
        lstm_model=load_model(os.path.join(models_folder,"lstm.keras"))
        lstm_input=scaled_test_data[:window_size] 
        print("Ok")
        
        print("Starting LSTM on CPU benchmark")
        for i in range(window_size,len(scaled_test_data)):
            # preprocessing
            lstm_input=scaled_test_data[i-window_size:i]
            x=lstm_input.reshape(1,window_size,1)
            
            # performing inference
            print(f"LSTM-cpu inference step {i}:")
            start_time=time.time_ns()
            prediction=lstm_model.predict(x)
            end_time=time.time_ns()
            print("Ok")

            # postprocessing
            inference_latency=end_time-start_time
            
            # updating results
            if(output): 
                results.append(["lstm_cpu",inference_latency,prediction.flatten().tolist()[0],scaled_test_data.flatten().tolist()[i]])        
        
    if(target_model=="sarima" or target_model=="all"):
        print("Sleeping 10 seconds before starting the benchmark:",end=" ")
        time.sleep(10)
        print("Ok")
        
        # loading sarima model
        print("Loading SARIMA model:",end=" ")
        sarima_model=ARIMAResults.load(os.path.join(models_folder,"sarima.pkl"))
        sarima_model=sarima_model.extend(scaled_test_data[:window_size])
        sarima_index=window_size
        print("Ok")

        for i in range(window_size,len(scaled_test_data)):
            # performing inference
            print(f"SARIMA inference step {i}:", end=" ")
            start_time=time.time_ns()
            prediction=sarima_model.forecast(steps=1)
            end_time=time.time_ns()
            inference_latency=end_time-start_time
            print("Ok")
        
            # postprocessing
            sarima_model=sarima_model.extend(scaled_test_data[i])
            
            # updating results
            if(output):
                results.append(["sarima",inference_latency,prediction.flatten().tolist()[0],scaled_test_data.flatten().tolist()[i]])    
    
    if(output):
        print("Saving latency results:",end=" ")
        results_df=pd.DataFrame(columns=["target_model","inference_latency_ns","predicted_value","real_value"],data=results)
        results_df.to_csv(output_file)
        print("Ok")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='Time-Series prediction latency benchmarker',
                    description='Python utility to benchmark different time-series prediction models.')

    parser.add_argument('-t','--target-model',
                        dest="target_model",
                        default='all',
                        choices=['all','lstm_cpu','cnn1d_cpu','sarima'],
                        help='time-series prediction model to be used for the benchmark')
    parser.add_argument('-o','--output-file',
                        metavar='OUTPUT_FILE',
                        dest="output_file",
                        default='latency-benchmark.csv',
                        help='file in which the output will be saved')
    parser.add_argument('--no-output',
                        dest="output",
                        action="store_false",
                        help='if set the command will save the benchmark result in a csv output file with the name specified with the -o option')
    
    
    args = parser.parse_args()
    main(args.target_model,args.output,args.output_file)
