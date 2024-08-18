from img_xtend.data.build import load_inference_source

source = "0"
source = "/dev/video0"
# source = "Jake"

a = load_inference_source(source)
print(a.source_type)
print(len(a))
print(a)
for i,elem in enumerate(a):
    print(type(elem))
    print(elem)
    if i == 10:
        break