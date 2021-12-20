import karnak3.core.log as kl
import karnak3.core.profiling as kprof

_logger = kl.logger('karnak.test.log')


def test_logs():
    kl.init()
    kl.set_level('DEBUG')
    kl.debug('debug')
    kl.debug('debug %s', 'parameterized')
    kl.trace('bad trace')
    kl.debug('debug-mem', memory=True)
    kl.error('error')
    kl.critical('critical')
    try:
        1 / 0
    except Exception as e:
        kl.exception('exception caught 1', exc_info=e)
        kl.exception('exception caught 2')
    _logger.log(0, 'trace-local')
    kl.trace('trace-local2')
    _logger.debug('debug-local')
    _logger.info('info-local')
    _logger.warning('warn-local')
    # kl.logger().setLevel(1)
    kl.set_level(0)
    kl.warning(f'loglevel {kl.logger().level}')
    kl.trace('good trace')
    kl.debug('debug')

    klogger1 = kl.KLog('karnak.test.log')
    klogger2 = kl.KLog('karnak.test.log2', level='INFO')
    klogger1.debug('klogger1 debug - good')
    klogger2.debug('klogger2 debug - bad')
    klogger2.info('klogger2 debug - good')

    klogger1.info('info-mem', memory=True)


def test_kprof():

    def waste_mem() -> dict:
        d = {}
        for i in range(1_000_000):
            d[i] = 'sadfkjlhsdfkjldsflkjg dfglkjdfsg hçsd fghjsd ghçdgh jfdjsf jk' + str(i)
        return d

    kp = kprof.KProfiler()
    kp.log_mem('kprof start')
    x = waste_mem()
    kp.log_delta('kprof delta')
    y = waste_mem()
    kp.log_cumulative('kprof cumulative')
    kp.log_cumulative('kprof cumulative delta only', delta_only=True)


if __name__ == "__main__":
    test_logs()
    test_kprof()

