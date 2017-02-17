from heapq import *

heap = []
data = [(10,"ten"), (3,"three"), (5,"five"), (7,"seven"), (9, "nine"), (2,"two")]
for item in data:
    heappush(heap, item)

#print heap.pop()

sorted = []
while heap:
    sorted.append(heappop(heap))
print sorted
data.sort()
print data == sorted

#print heap.pop()