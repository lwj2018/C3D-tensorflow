def un_avg_pool(name, l_input, k, images_placeholder, test_images):
  """
    the input is an 5-D array: batch_size length width height channels
  """
  input_shape = l_input.get_shape().as_list()
  batch_size = input_shape[0]
  length = input_shape[1]
  width = input_shape[2]
  height = input_shape[3]
  channels = input_shape[4]
  output_shape = [batch_size,k*length,2*width,2*height,channels]

  sess.run(tf.global_variables_initializer())
  input_array = np.zeros(input_shape,dtype = np.float32)
  output_array = np.zeros(output_shape,dtype = np.float32)
  input_array = l_input.eval(session = sess,feed_dict={images_placeholder:test_images})
  #input_array = np.array(l_input.as_list())

  for n in range(batch_size):
    for l in range(k*length):
      for w in range(2*width):
        for h in range(2*height):
          for c in range(channels):
            output_array[n,l,w,h,c] = input_array[n,int(l/k),int(w/2),int(h/2),c]
  output = tf.convert_to_tensor(output_array)
  return output

def un_max_pool(name, l_input, l_output, k, images_placeholder, test_images):
  '''
  parameters:
    l_input is the input of pool
    l_output is the output of pool
    according to input,we can get the max index
    according to output and max_index, unpool and reconstruct the input 
  return:
    the reconstructed input
  '''
  input_shape = l_input.get_shape().as_list()
  output_shape = l_output.get_shape().as_list()
  batch_size = output_shape[0]
  length = output_shape[1]
  rows = output_shape[2]
  cols = output_shape[3]
  channels = output_shape[4]
  input_array = l_input.eval(session = sess,feed_dict={images_placeholder:test_images})
  output_array = l_output.eval(session = sess,feed_dict = {images_placeholder:test_images})
  unpool_array = np.zeros(input_shape,dtype = np.float32)
  for n in range(batch_size):
    for l in range(length):
      for r in range(rows):
        for c in range(cols):
          for ch in range(channels):
            l_in, r_in, c_in = k*l, 2*r, 2*c
            sub_square = input_array[ n, l_in:l_in+k, r_in:r_in+2, c_in:c_in+2, ch ]
            max_pos_l, max_pos_r, max_pos_c = np.unravel_index(np.nanargmax(sub_square), (k, 2, 2))
            array_pixel = output_array[ n, l, r, c, ch ]
            unpool_array[n, l_in + max_pos_l, r_in + max_pos_r, c_in + max_pos_c, ch] = array_pixel
  unpool = tf.convert_to_tensor(unpool_array)

  return unpool

def un_max_pool_approximate(name,l_input,output_shape,k):
  # out = tf.concat([l_input,tf.zeros_like(l_input)], 4) #concat along the channel axis
  # out = tf.concat([out,tf.zeros_like(out)], 3) #concat along the no.-2 axis
  # if k == 2:
  #   out = tf.concat([out,tf.zeros_like(out)], 2) #concat along the no.-3 axis
  # out = tf.reshape(out,output_shape)
  out = tf.concat([l_input,l_input], 4) #concat along the channel axis
  out = tf.concat([out,out], 3) #concat along the no.-2 axis
  if k == 2:
    out = tf.concat([out,out], 2) #concat along the no.-3 axis
  out = tf.reshape(out,output_shape)
  return out