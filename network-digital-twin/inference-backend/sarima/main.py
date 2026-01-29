import inference_pb2
import inference_pb2_grpc
import os
from datetime import datetime
import pickle
from statsmodels.tsa.arima.model import ARIMAResults
from influxdb_client_3 import InfluxDBClient3, write_client_options, SYNCHRONOUS
import grpc
from concurrent import futures



INFLUX_HOST=os.getenv("INFLUX_HOST")
INFLUX_DB=os.getenv("INFLUX_DB")
NETWORK_CELL=os.getenv("NETWORK_CELL")


class SARIMAInferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self):
        super().__init__()

        print("Loading scaler:",end=" ")
        self.scaler=pickle.load(open("scaler.pkl","rb"))
        print("Ok")

        print("Loading SARIMA model:",end=" ")
        self.sarima_model=ARIMAResults.load("sarima.pkl")
        self.trained_until=datetime.now() # TODO change with the actual date used in the training set, not the one of the object init
        print("Ok")

        print("Creating InfluxDB client:",end=" ")
        wco = write_client_options(write_options=SYNCHRONOUS)
        self.influx_client = InfluxDBClient3(
            host=f"{INFLUX_HOST}",
            database=f"{INFLUX_DB}",
            # token=f"{INFLUX_TOKEN}",
            write_client_options=wco
            )
        print("Ok")

    def make_inference(self, request, context):
        request_time=datetime.fromtimestamp(int(request.datetime.seconds))
        print(f"Received inference request for datetime: {request_time}")

        print("Querying InfluxDB:",end=" ")
        query_result = self.influx_client.query(f"""
                                                SELECT *
                                                FROM requests
                                                WHERE time >= to_timestamp('{self.trained_until}')
                                                AND time <= to_timestamp('{request_time}')
                                                ORDER BY time DESC
                                                """,mode="pandas")
        print("Ok")

        print("Preprocessing input: ",end=" ")
        x= self.scaler.transform(query_result[["requests"]].values)

        print("Ok")

        print(f"Performing inference:")
        y_pred=self.sarima_model.extend(x).forecast(step=1).flatten().tolist()[0]
        print("Ok")

        return inference_pb2.InferenceResult(prediction=y_pred)
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(
        SARIMAInferenceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()