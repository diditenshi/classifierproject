import streamlit as st

st.set_page_config(
    page_title="Banana Disease Detection Web Application",
    page_icon="🍌",
)

st.write("# Banana _(Musa sp.)_ 👋")
st.success(
    """_Notes (Reprinted from Philippine Council for Agriculture, Aquatic and Natural Resources Research and Development.
    Banana production manual. Los Baños, Laguna: PCAARRD-DOST, 2013. 129p. -(PCAARRD Book Series No. 175-A/2013; Reprint)_
    """)


st.markdown(
    """
    Bananas are one of the most important fruit crop in the Philippines, responsible for contributing hundreds of million dollars
    annually due to exports on fresh fruits, processed bananas such as chips, crackers and catsup. There are eighty distinct banana 
    cultivars grown in the country but aside from the exported cultivar, "Cavendish", there are other cultivars such as "Lakatan", "Latundan" and
    "Buñgulan", "Saba" and "Señorita" are being cultivated. Diseases in crops can cause major loss to the farmers due to low yields as it may affect 
    the crop growth and the quality of the fruit that the crops may produce when infested. In order to avoid such problems, timely identification of 
    the diseases and the application of an effective disease management method will help the farmers to control and manage diseases as well as reduce 
    its harmful effects if not eradicate.

    The following are the identified major diseases of bananas:
    
    ### 🍀 Major Diseases of Banana 🍀
    **🔸 Moko Disease (Bacterial Wilt)**
    - **Causal Organism:** Bacterium (_Pseudomonas solanacearum_ B.F. Smith) 
    - **Description:** This bacterium has four major strains (strains D, B, SFR, and I). Strains SFR and I are readily transmitted
      by insects visiting flowers, but these have low soil persistence. Strain B can persist in the soil for up to 18 months. Strain
      D infects _Heliconia_ spp. and has low virulence on banana.
    - **Symptoms:** Moko is a very contagious disease that can kill an infected banana in just a few weeks. At its initial stage, bacterial
      wilt develops as a yellowish coloration of the inner leaf lamina, close to the petiole. The inner leaves and in some cases, the heart leaf
      also collapses. A diseased trunk when cut traversely shows that all vascular strands are discolored ranging from pale yellow
      to dark brown or bluish-black. Presence of bacterial exudates with a dirty-white and/or brown color can be observed. Fruits
      appear distorted yellow or dark brown. The disease symptoms are manifested within ten days or more after the entry of the bacterium
      into the plant.

    **🔸 Panama Disease (Fusarium Wilt)**
    - **Causal Organism:** Fungus (_Fusarium oxysporum_ f. sp. _cubense_ [E.F. Smith])
    - **Description:** The soil-borne fungus enters only through the roots and grows and then sporulates abundantly in the xylem
      vessels until it invades and blocks the whole vascular system, making the plant to wilt. The most common means of spreading
      the pathogen is through the infected rhizome but the fungus can also spread in the soil, running water and from the farm implements.
    - **Symptoms:** All the leaves will turn yellow and continues to collapse and dry up until eventually, all the leaves will wilt.
      The pseudostem may crack at the base but it will remain standing until it decays and falls. When cut, the pseudostem shows
      brown to purple discoloration of the vascular tissues with reddish tinge.

    **🔸 Bunchy Top Disease**
    - **Causal Organism:** banana bunchy top virus
    - **Description:** The banana aphid _Pentalonia nigronervosa_ Coq. feeds on the virus-infected plant and transmits the virus
      into the healthy plant. When the plant is infected, the virus multiplies and moves to all parts of the plant. 
    - **Symptoms:** The affected plant shows green streaks on the secondary veins on the underside of the lamina and on the midrib 
      and petiole which varies from dark green dots to continuous dark-green line with a ragged edge. The plant is dwarfed and shows
      marginal chlorosis and curling. As the disease progresses, the subsequent leaves unfurl prematurely but slowly which results to
      smaller and stunted leaves on the crown of the plant. It seldom bear fruit and if they do, the hands are deformed and the fingers 
      are reduced.

    **🔸 Sigatoka Disease (Black leaf streak)**
    - **Causal Organism:** Fungus (_Mycosphaerella fijiensis_)
    - **Description:** The fungus produces two (2) types of microscopic spores (seeds) called conidia and ascopores. When the spores
      land on the leaf, they germinate when covered with a film of water. The fungus grows over the leaf surface for several days
      before it enters the stomata and infects the cells in the leaf.
    - **Symptoms:** Tiny brown streaks appear around the stomata a few weeks after infection. The streaks enlarge, turn blackish, and later
      on become brown oval spots with yellow margins.These streaks will group together causing the leaf to turn black and dry up prematurely.

"""
)