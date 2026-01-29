import inference_pb2
import inference_pb2_grpc
from influxdb_client_3 import InfluxDBClient3, write_client_options, SYNCHRONOUS
import datetime
import grpc

min_datetime=1385074800+24*3600
max_datetime=1385848800
with grpc.insecure_channel('localhost:50051') as channel:
    stub = inference_pb2_grpc.InferenceStub(channel)
    for timestamp in range(min_datetime,max_datetime,3600):
        request_date=inference_pb2.InferenceRequest(datetime=datetime.datetime.fromtimestamp(timestamp))
        response = stub.make_inference(request_date)
        print(f"Inference result for '{datetime.datetime.fromtimestamp(min_datetime)}': {response.prediction}")
