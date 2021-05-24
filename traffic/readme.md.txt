first training result
accuracy 14% 
layers=[conv2d(10,),maxpool,conv2d(5),maxpool,flatten,dense(256),dense(128),dense(64),softmax(43)


2nd training result
accuracy 21% 
layers=[conv2d(8,),maxpool,conv2d(64),maxpool,conv2d(256),flatten,dense(256),dense(128),dense(64),softmax(43)

then I divided input images values by 255.0
layers=[conv2d(32,),maxpool,conv2d(64),maxpool,conv2d(256),flatten,dense(256),dense(128),dense(64),softmax(43)]
and got an accuracy of 28.38%


3rd training
layers=[conv2d(32),conv2d(64), maxpool, conv2d(128), conv2d(256), maxpool, flatten,
	dense(256), dropout, dense(128), dropout, dense(64), dropout), dense(32), 
	softmax]

epochs=100
training accuracy obtained=57% because of memory exhaustion


4th training
layers=[conv2d(100),conv2d(200), maxpool, conv2d(400), conv2d(800), maxpool, flatten,
	dense(256), dropout, dense(128), dropout, dense(64), dropout), dense(32), 
	softmax]

epochs=100
training accuracy obtained=56% training failed

then in at last I trained the models on colab notebook
with the given model
layers=[conv2d(32), maxpool, conv2d(128),  maxpool, conv2d(256), maxpool,
	dense(1024), dropout(0.5), dense(256),dropout(.2), dense(64), dropout(.2), softmax(43)]
for each layers after the flatten kernel regularizer with l2 was used.

and for conv2d layers kernel size was (3,3), activation=relu 
for maxpool pool size was (2,2)
test accuracy of 37% 










