
import torch

from utils import (
    RDF_NUMBER_TYPES,
    get_relevant_relations,
    ALL_LITERALS,
    POTENTIAL_TEXT_TYPES,
    RDF_DATE_TYPES,
    IMAGE_TYPES,
    GEO_TYPES,
    delete_r,
    Data
)






def delete_all_literals(data: Data, **kwargs) -> Data:
    """
    Delete all literal values from the dataset.

    This method takes a Data object representing a KG and deletes all literal values from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all literal values deleted.

    """
    rr = get_relevant_relations(data, ALL_LITERALS)
    return delete_r(data, torch.tensor(rr))


def delete_all_but_numbers(data: Data, **kwargs) -> Data:
    """
    Delete all literal values but numbers from the dataset.

    This method takes a Data object representing a KG and deletes all literal values but numbers from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all literal values but numbers deleted.

    """
    to_exclude = RDF_NUMBER_TYPES
    rr = get_relevant_relations(
        data, [l for l in ALL_LITERALS if l not in to_exclude]
    )
    return delete_r(data, torch.tensor(rr))


def delete_all_but_dates(data: Data, **kwargs) -> Data:
    """
    Delete all literal values but dates from the dataset.

    This method takes a Data object representing a KG and deletes all literal values but dates from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all literal values but dates deleted.

    """
    to_exclude = RDF_DATE_TYPES
    rr = get_relevant_relations(
        data, [l for l in ALL_LITERALS if l not in to_exclude]
    )
    return delete_r(data, torch.tensor(rr))


def delete_all_but_text(data: Data, **kwargs) -> Data:
    """
    Delete all literal values but numbers from the text.

    This method takes a Data object representing a KG and deletes all literal values but text from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all literal values but text deleted.

    """
    to_exclude = POTENTIAL_TEXT_TYPES
    rr = get_relevant_relations(
        data, [l for l in ALL_LITERALS if l not in to_exclude]
    )
    return delete_r(data, torch.tensor(rr))


def delete_all_but_images(data: Data, **kwargs) -> Data:
    """
    Delete all literal values but numbers from the images.

    This method takes a Data object representing a KG and deletes all literal values but images from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all literal values but images deleted.

    """
    to_exclude = IMAGE_TYPES
    rr = get_relevant_relations(
        data, [l for l in ALL_LITERALS if l not in to_exclude]
    )
    return delete_r(data, torch.tensor(rr))


def delete_number_literals(data: Data, **kwargs) -> Data:
    """
    Delete all number literal values from the dataset.

    This method takes a Data object representing a KG and deletes all number literal values from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all number literal values deleted.

    """
    rr = get_relevant_relations(data, RDF_NUMBER_TYPES)
    return delete_r(data, torch.tensor(rr))


def delete_date_literals(data: Data, **kwargs) -> Data:
    """
    Delete all date literal values from the dataset.

    This method takes a Data object representing a KG and deletes all date literal values from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all date literal values deleted.

    """
    rr = get_relevant_relations(data, RDF_DATE_TYPES)
    return delete_r(data, torch.tensor(rr))


def delete_image_literals(data: Data, **kwargs) -> Data:
    """
    Delete all image literal values from the dataset.

    This method takes a Data object representing a KG and deletes all image literal values from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all image literal values deleted.

    """
    rr = get_relevant_relations(data, IMAGE_TYPES)
    return delete_r(data, torch.tensor(rr))


def delete_text_literals(data: Data, **kwargs) -> Data:
    """
    Delete all text literal values from the dataset.

    This method takes a Data object representing a KG and deletes all text literal values from the dataset.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments for customization or specific deletion options.

    Returns:
        Data: The updated Data object with all text literal values deleted.

    """
    rr = get_relevant_relations(data, POTENTIAL_TEXT_TYPES)
    return delete_r(data, torch.tensor(rr))


def delete_geo_literals(data: Data, **kwargs) -> Data:
    rr = get_relevant_relations(data, GEO_TYPES)
    return delete_r(data, torch.tensor(rr))


def delete_none_literals(data: Data, **kwargs) -> Data:
    rr = get_relevant_relations(data, POTENTIAL_TEXT_TYPES)
    return delete_r(data, torch.tensor(rr))
