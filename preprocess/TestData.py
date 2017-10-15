import Net

def run(numSamples=10):
   batcher, _ = Net.dataBatcher(numSamples, numSamples, human=False)
   dat = batcher.next()
   batcher.vis(dat)

