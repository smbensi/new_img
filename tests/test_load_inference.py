from img_xtend.data.build import load_inference_source

source = "0"
# source = "Jake"

a = load_inference_source(source)
print(a.source_type)
print(len(a))
print(a)
# for elem in a:
#     print(type(elem))
#     print(elem)
