from img_xtend.data.build import load_inference_source

source = "0"

a = load_inference_source(source)
print(a.source_type)