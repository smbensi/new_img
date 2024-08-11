from img_xtend.data.build import load_inference_source

source = "Jake/2022-01-23-180541_1.jpg"

a = load_inference_source(source)
print(a.source_type)