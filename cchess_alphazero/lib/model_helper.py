import os
from logging import getLogger

logger = getLogger(__name__)


def _fresh_start_enabled(model):
    return bool(getattr(model.config.opts, "new", False))


def _best_model_paths(model):
    rc = model.config.resource
    return rc.model_best_config_path, rc.model_best_weight_path


def _sl_best_model_paths(model):
    rc = model.config.resource
    return rc.sl_best_config_path, rc.sl_best_weight_path


def load_best_model_weight(model):
    """
    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    if _fresh_start_enabled(model):
        logger.info("Fresh-start mode active; skip loading BestModel checkpoint.")
        return False
    config_path, weight_path = _best_model_paths(model)
    return model.load(config_path, weight_path)


def load_best_model_weight_from_internet(model):
    """
    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    if _fresh_start_enabled(model):
        logger.info("Fresh-start mode active; skip downloading BestModel checkpoint.")
        return False
    from cchess_alphazero.lib.web_helper import download_file
    logger.info("download model from remote server")
    download_file(model.config.internet.download_url, model.config.resource.model_best_weight_path)
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def save_as_best_model(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)


def build_fresh_best_model(model):
    logger.info("Initialize a fresh random BestModel.")
    model.build()
    save_as_best_model(model)
    return model


def need_to_reload_best_model_weight(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    if _fresh_start_enabled(model):
        logger.debug("Fresh-start mode active; skip BestModel reload checks.")
        return False
    logger.debug("start reload the best model if changed")
    digest = model.fetch_digest(model.config.resource.model_best_weight_path)
    if digest != model.digest:
        return True

    logger.debug("the best model is not changed")
    return False


def load_model_weight(model, config_path, weight_path, name=None):
    if name is not None:
        logger.info(f"{name}: load model from {config_path}")
    return model.load(config_path, weight_path)


def save_as_next_generation_model(model):
    return model.save(model.config.resource.next_generation_config_path, model.config.resource.next_generation_weight_path)


def load_sl_best_model_weight(model):
    """
    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    if _fresh_start_enabled(model):
        logger.info("Fresh-start mode active; skip loading SL BestModel checkpoint.")
        return False
    config_path, weight_path = _sl_best_model_paths(model)
    return model.load(config_path, weight_path)


def save_as_sl_best_model(model):
    """

    :param cchess_alphazero.agent.model.CChessModel model:
    :return:
    """
    return model.save(model.config.resource.sl_best_config_path, model.config.resource.sl_best_weight_path)


def build_fresh_sl_best_model(model):
    logger.info("Initialize a fresh random SL BestModel.")
    model.build()
    save_as_sl_best_model(model)
    return model


def next_generation_model_exists(config):
    rc = config.resource
    return os.path.exists(rc.next_generation_config_path) and os.path.exists(rc.next_generation_weight_path)


def is_next_generation_model_fresh(config):
    if not next_generation_model_exists(config):
        return False
    if not getattr(config.opts, "new", False):
        return True

    rc = config.resource
    if not (os.path.exists(rc.model_best_config_path) and os.path.exists(rc.model_best_weight_path)):
        return False

    best_timestamp = max(os.path.getmtime(rc.model_best_config_path), os.path.getmtime(rc.model_best_weight_path))
    next_generation_timestamp = min(
        os.path.getmtime(rc.next_generation_config_path),
        os.path.getmtime(rc.next_generation_weight_path),
    )
    return next_generation_timestamp >= best_timestamp
