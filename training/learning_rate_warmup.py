import tensorflow as tf

# The base learning rate is what will be modified in the training process to achieve 
# the piecewise constant part. The class itself actually only includes linear warm up 
# and a constant learning rate afterwards.

# Base learning rate and target learning rate should both be set as the rate you want the warm up to end with.

class LinearWarmupToPiecewiseConstant(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr_var, initial_lr, target_lr, warmup_steps):
        super().__init__()
        self.base_lr_var = base_lr_var 
        self.initial_lr = tf.constant(initial_lr, tf.float32)
        self.target_lr  = tf.constant(target_lr,  tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, tf.int64)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        wus_f  = tf.cast(self.warmup_steps, tf.float32)
        frac = tf.minimum(1.0, step_f / tf.maximum(1.0, wus_f))
        warmup_lr = self.initial_lr + frac * (self.target_lr - self.initial_lr)

        # If the step number is still in the warm up regime, then use the warmup learning rate.
        # Else, use the base learning rate.

        # If warm up steps = 0, we start with base learning rate directly.
        return tf.where(step_f < wus_f, warmup_lr, tf.cast(self.base_lr_var, tf.float32))
    
    def get_config(self):
        return {
            "base_lr_var" : self.base_lr_var,
            "initial_lr"  : self.initial_lr,
            "target_lr"   : self.target_lr,
            "warmup_steps": self.warmup_steps
        }