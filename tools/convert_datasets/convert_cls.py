import json

fname = "/root/autodl-tmp/DAFormer/data/cityscapes/sample_class_stats_dict.json"
outname = "/root/autodl-tmp/DAFormer/data/cityscapes/cls_label.json"
f = open(fname)

data = json.load(f)

dict = {}

i = 0
for img in data:
    label = [0] * 19
    for idx in data[img]:
        label[int(idx)] = 1
    abs_img = "/root/autodl-tmp/DAFormer/" + img[:9] + "images" + img[15:21] + ".png"
    dict[abs_img] = label
    print(i)
    i += 1

with open(outname, "w") as outfile:
    json.dump(dict, outfile)

f.close()