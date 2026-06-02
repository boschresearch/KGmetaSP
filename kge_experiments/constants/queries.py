# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# MLSEAKG_FILTER_QUERY = """
# PREFIX mls: <http://www.w3.org/ns/mls#>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX dcterms: <http://purl.org/dc/terms/>
# PREFIX dcat: <http://www.w3.org/ns/dcat#>
# PREFIX prov: <http://www.w3.org/ns/prov#>
# PREFIX mlso: <http://w3id.org/mlso/>
# PREFIX schema: <http://schema.org/>
# PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
# PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

# CONSTRUCT {{
#   ?dataset rdfs:label ?datasetLabel ;
#            dcterms:title ?datasetTitle ;
#            dcterms:description ?datasetDescription ;
#            dcterms:license ?datasetLicense ;
#            dcat:landingPage ?datasetLandingPage ;
#            dcterms:accessRights ?datasetAccessRights ;
#            dcat:distribution ?distribution .
#   ?distribution mls:hasQuality ?datasetQuality ;
#                mls:hasPart ?feature ;
#                mlso:hasDefaultTargetFeature ?targetFeature .
#   ?datasetQuality rdfs:label ?datasetQualityLabel ;
#                   mlso:hasDataCharacteristicType ?datasetDataType ;
#                   mls:hasValue ?datasetQualityValue .
#   ?feature rdfs:label ?featureLabel ;
#            dcterms:title ?featureTitle ;
#            mls:hasQuality ?featureQuality .
#   ?featureQuality rdfs:label ?qualityLabel ;
#                   mlso:hasDataCharacteristicType ?dataType ;
#                   mls:hasValue ?qualityValue .
# }}
# WHERE {{
#   VALUES ?datasetID {{ {values_clause} }}
#   ?dataset a mls:Dataset ;
#            dcterms:identifier ?datasetID .
#   OPTIONAL {{ ?dataset rdfs:label ?datasetLabel }}
#   OPTIONAL {{ ?dataset dcterms:title ?datasetTitle }}
#   OPTIONAL {{ ?dataset dcterms:description ?datasetDescription }}
#   OPTIONAL {{ ?dataset dcterms:license ?datasetLicense }}
#   OPTIONAL {{ ?dataset dcat:landingPage ?datasetLandingPage }}
#   OPTIONAL {{ ?dataset dcterms:accessRights ?datasetAccessRights }}
#   OPTIONAL {{
#     ?dataset dcat:distribution ?distribution .
#     OPTIONAL {{
#       ?distribution mls:hasQuality ?datasetQuality .
#       OPTIONAL {{ ?datasetQuality rdfs:label ?datasetQualityLabel }}
#       OPTIONAL {{ ?datasetQuality mlso:hasDataCharacteristicType ?datasetDataType }}
#       OPTIONAL {{ ?datasetQuality mls:hasValue ?datasetQualityValue }}
#     }}
#     OPTIONAL {{ ?distribution mlso:hasDefaultTargetFeature ?targetFeature }}
#   }}
#   OPTIONAL {{
#     ?distribution mls:hasPart ?feature
#     OPTIONAL {{
#         ?feature mls:hasQuality ?featureQuality .
#         OPTIONAL {{ ?featureQuality rdfs:label ?qualityLabel }}
#         OPTIONAL {{ ?featureQuality mlso:hasDataCharacteristicType ?dataType }}
#         OPTIONAL {{ ?featureQuality mls:hasValue ?qualityValue }}
#     }}
#     OPTIONAL {{ ?feature rdfs:label ?featureLabel }}
#     OPTIONAL {{ ?feature dcterms:title ?featureTitle }}
#   }}
# }}
# """

MLSEAKG_FILTER_QUERY = """
PREFIX mls: <http://www.w3.org/ns/mls#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX mlso: <http://w3id.org/mlso/>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

CONSTRUCT {{
  ?dataset rdfs:label ?datasetLabel ;
           dcterms:title ?datasetTitle ;
           dcterms:description ?datasetDescription ;
           dcterms:license ?datasetLicense ;
           dcat:landingPage ?datasetLandingPage ;
           dcterms:accessRights ?datasetAccessRights ;
          #  mls:hasPart ?feature ;
           dcat:distribution ?distribution .
  ?distribution mls:hasQuality ?datasetQuality ;
               mls:hasPart ?feature .
              #  mlso:hasDefaultTargetFeature ?targetFeature .
  ?datasetQuality rdfs:label ?datasetQualityLabel ;
                  mlso:hasDataCharacteristicType ?datasetCharacteristicType ;
                  mls:hasValue ?datasetQualityValue .
  ?feature rdfs:label ?featureLabel;
           dcterms:title ?featureTitle;
                  #  ex:datatype ?featureDataType;
           mls:hasQuality ?featureQuality .
  ?featureQuality rdfs:label ?qualityLabel ;
                  mlso:hasDataCharacteristicType ?characteristicType ;
                  mls:hasValue ?qualityValue .
}}
WHERE {{
    VALUES ?datasetID {{ {values_clause} }}
    ?dataset a mls:Dataset ;
             dcterms:identifier ?datasetID ;
             dcat:distribution ?distribution .
    OPTIONAL {{ ?dataset rdfs:label ?datasetLabel }}
    OPTIONAL {{ ?dataset dcterms:title ?datasetTitle }}
    OPTIONAL {{ ?dataset dcterms:description ?datasetDescription }}
    OPTIONAL {{ ?dataset dcterms:license ?datasetLicense }}
    OPTIONAL {{ ?dataset dcat:landingPage ?datasetLandingPage }}
    OPTIONAL {{ ?dataset dcterms:accessRights ?datasetAccessRights }}

    ?distribution mls:hasPart ?feature .
    OPTIONAL {{
      ?distribution mls:hasQuality ?datasetQuality .
      ?datasetQuality mlso:hasDataCharacteristicType ?datasetCharacteristicType ;
                      mls:hasValue ?datasetQualityValue .
      OPTIONAL {{ ?datasetQuality rdfs:label ?datasetQualityLabel }}
    }}
    # OPTIONAL {{ ?distribution mlso:hasDefaultTargetFeature ?targetFeature }}

    ?feature rdfs:label ?featureLabel ;
             mls:hasQuality ?featureQuality .
    OPTIONAL {{ ?feature dcterms:title ?featureTitle }}

    ?featureQuality mlso:hasDataCharacteristicType ?characteristicType ;
                    mls:hasValue ?qualityValue .
    OPTIONAL {{ ?featureQuality rdfs:label ?qualityLabel }}
}}
"""

MLSEAKG_FEATURE_IRIS = """
PREFIX mls: <http://www.w3.org/ns/mls#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX mlso: <http://w3id.org/mlso/>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT
  ?feature
WHERE {{
    VALUES ?datasetID {{ {values_clause} }}
    ?dataset a mls:Dataset ;
             dcterms:identifier ?datasetID ;
             dcat:distribution ?distribution .
    ?distribution mls:hasPart ?feature .
}}
"""
