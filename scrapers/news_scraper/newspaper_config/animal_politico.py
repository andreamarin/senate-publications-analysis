BASE_URL = "https://animalpolitico.com/"
NEWSPAPER_NAME = "animal_politico"
BATCH_SIZE = 20

# sections data
SECTIONS = {
    "politica": "Poltica",
    "salud": "Salud",
    "seguridad": "Seguridad",
    "genero_y_diversidad": "GneroYDiversidad",
    "sociedad": "Sociedad",
    "estados": "Estados",
    "tendencias": "AnimalMX",
    "analisis": "NotaDePlumaje",
    "internacional": "Internacional",
    "hablemos_de": "HablemosDe"
}

EXTRA_QUERY = {
    "hablemos_de": """
    categorasDeHablemosDe {
        edges {
            node {
                id
                slug
            }
        }
    }
    """,
    "analisis": """
    blogSlug
    blogAuthor
    """
}

OP_SECTION_NAME = {
    "politica": "Politica"
}

KEYS_NAMES = {
    "analisis": "notasDePlumaje"
}

SUBCATEGORIES = {
    "hablemos_de": {"finanzas", "empresas", "sustentabilidad", "educacion"}
}

KEYS_MAPPING = {
    "databaseId": "id",
    "title": "title",
    "postExcerpt": "summary",
}

# search config
HEADERS = {
  'Content-Type': 'application/json'
}
SEARCH_URL = "https://panel.animalpolitico.com/graphql"

OPERATION_NAME = "FetchAll{section}"
NEWS_KEY = "all{section}"
QUERY = """query {op_name}($where: RootQueryTo{section}ConnectionWhereArgs) {{
    {key}(where: $where) {{
        edges {{
            node {{
                databaseId
                title
                slug
                contentRendered
                categoryPrimarySlug
                postExcerpt
                date
                terms {{
                    edges {{
                        node {{
                            id
                            slug
                        }}
                    }}
                }}
                {extra_query}
            }}
        }}
        pageInfo {{
            offsetPagination {{
                total
            }}
        }}
    }}
}}"""