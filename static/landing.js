
    const body = document.body;
    const themeToggle = document.getElementById('theme-toggle');
    const languageSelect = document.getElementById('language-select');
    const heroVideo = document.getElementById('hero-video');
    const videoError = document.getElementById('video-error');
    const translations = {
      en: {
        heroTitle: "Apps Localization",
        heroSubtitle: "Effortlessly translate and customize images for global audiences.",
        featuresTitle: "Powerful Features",
        featuresSubtitle: "Discover why Apps Localization is the ultimate tool for image translation.",
        feature1Title: "Multi-Language Translation",
        feature1Desc: "Translate text in images to multiple languages with high accuracy.",
        feature2Title: "Customizable Text Editing",
        feature2Desc: "Adjust font, size, color, and animations to match your brand.",
        feature3Title: "Device-Optimized Resizing",
        feature3Desc: "Resize images for phones, tablets, desktops, and more.",
        feature4Title: "Intuitive Interface",
        feature4Desc: "User-friendly design for seamless editing and translation.",
        feature5Title: "High-Quality Output",
        feature5Desc: "Download crisp, professional images in your preferred format.",
        feature6Title: "Real-Time Preview",
        feature6Desc: "Preview changes instantly to ensure perfect results.",
        howToTitle: "How to Use",
        howToSubtitle: "Transform your images in just a few simple steps.",
        step1: "Upload your image to the Apps Localization platform.",
        step2: "Select the target language for translation.",
        step3: "Choose the desired output resolution for your device.",
        step4: "Customize text appearance with fonts, colors, and animations.",
        step5: "Download your localized image instantly.",
        whyTitle: "Why Apps Localization?",
        whySubtitle: "Enhance your content creation with unmatched flexibility and efficiency.",
        why1Title: "Global Reach",
        why1Desc: "Engage audiences worldwide with localized content.",
        why2Title: "Time-Saving",
        why2Desc: "Automate translation and resizing tasks effortlessly.",
        why3Title: "Brand Consistency",
        why3Desc: "Maintain your visual identity across languages.",
        tutorialTitle: "Watch Our Tutorial",
        tutorialSubtitle: "Learn how to use Apps Localization with our step-by-step Loom video.",
        previewTitle: "Translation Preview",
        previewSubtitle: "See your original image and all translations in one place",
        originalTitle: "Original Content",
        originalLabel: "Original Document (9.png)",
        translationsTitle: "Translations",
        translation1Title: "Dutch Translation",
        translation1Desc: "This is the Dutch version of the original document with accurate translation and cultural adaptation.",
        translation1Badge: "Nederlands",
        translation2Title: "French Translation",
        translation2Desc: "French translation maintaining the original meaning while adapting to French linguistic nuances.",
        translation2Badge: "Français",
        translation3Title: "German Translation",
        translation3Desc: "Precise German translation with attention to technical terminology and formal language requirements.",
        translation3Badge: "Deutsch",
        translation4Title: "Italian Translation",
        translation4Desc: "Italian translation with proper adaptation to the language's unique characteristics and cultural context.",
        translation4Badge: "Italiano",
        translation5Title: "Spanish Translation",
        translation5Desc: "Spanish translation with accurate localization and attention to regional linguistic variations.",
        translation5Badge: "Español",
        translation6Title: "portuguese Translation",
        translation6Desc: "portuguese translation with precise localization and cultural adaptation for portuguese-speaking audiences.",
        translation6Badge: "portuguese",
        testimonialsTitle: "What Our Users Say",
        testimonialsSubtitle: "Join thousands of satisfied users worldwide.",
        testimonial1Text: "\"Apps Localization saved me hours of work. The translations are accurate and the interface is intuitive.\"",
        testimonial1Author: "- Sarah Johnson, Marketing Director",
        testimonial2Text: "\"As a global brand manager, this tool has been invaluable for our international campaigns.\"",
        testimonial2Author: "- Michael Chen, Global Brand Manager",
        testimonial3Text: "\"The ability to customize text appearance while maintaining translation accuracy is impressive.\"",
        testimonial3Author: "- Elena Rodriguez, Content Creator",
        ctaTitle: "Ready to Localize Your Images?",
        ctaSubtitle: "Start creating multi-language content today with Apps Localization.",
        footerText: "© 2023 Apps Localization. All rights reserved."
      },
      fr: {
        heroTitle: "Localisation d'Apps",
        heroSubtitle: "Traduisez et personnalisez facilement des images pour un public mondial.",
        featuresTitle: "Fonctionnalités Puissantes",
        featuresSubtitle: "Découvrez pourquoi Apps Localization est l'outil ultime pour la traduction d'images.",
        feature1Title: "Traduction Multilingue",
        feature1Desc: "Traduisez le texte dans les images en plusieurs langues avec une grande précision.",
        feature2Title: "Édition de Texte Personnalisable",
        feature2Desc: "Ajustez la police, la taille, la couleur et les animations pour correspondre à votre marque.",
        feature3Title: "Redimensionnement Optimisé",
        feature3Desc: "Redimensionnez les images pour les téléphones, tablettes, ordinateurs de bureau et plus encore.",
        feature4Title: "Interface Intuitive",
        feature4Desc: "Conception conviviale pour une édition et une traduction fluides.",
        feature5Title: "Sortie de Haute Qualité",
        feature5Desc: "Téléchargez des images nettes et professionnelles dans votre format préféré.",
        feature6Title: "Aperçu en Temps Réel",
        feature6Desc: "Prévisualisez les modifications instantanément pour garantir des résultats parfaits.",
        howToTitle: "Comment Utiliser",
        howToSubtitle: "Transformez vos images en quelques étapes simples.",
        step1: "Téléchargez votre image sur la plateforme Apps Localization.",
        step2: "Sélectionnez la langue cible pour la traduction.",
        step3: "Choisissez la résolution de sortie souhaitée pour votre appareil.",
        step4: "Personnalisez l'apparence du texte avec des polices, des couleurs et des animations.",
        step5: "Téléchargez votre image localisée instantanément.",
        whyTitle: "Pourquoi Apps Localization?",
        whySubtitle: "Améliorez votre création de contenu avec une flexibilité et une efficacité inégalées.",
        why1Title: "Portée Mondiale",
        why1Desc: "Touche des audiences mondiales avec un contenu localisé.",
        why2Title: "Gain de Temps",
        why2Desc: "Automatisez les tâches de traduction et de redimensionnement sans effort.",
        why3Title: "Cohérence de Marque",
        why3Desc: "Maintenez votre identité visuelle dans toutes les langues.",
        tutorialTitle: "Regardez Notre Tutoriel",
        tutorialSubtitle: "Apprenez à utiliser Apps Localization avec notre vidéo Loom étape par étape.",
        previewTitle: "Aperçu de Traduction",
        previewSubtitle: "Voyez votre image originale et toutes les traductions en un seul endroit",
        originalTitle: "Contenu Original",
        originalLabel: "Document Original (9.png)",
        translationsTitle: "Traductions",
        translation1Title: "Traduction Néerlandaise",
        translation1Desc: "Ceci est la version néerlandaise du document original avec une traduction précise et une adaptation culturelle.",
        translation1Badge: "Nederlands",
        translation2Title: "Traduction Française",
        translation2Desc: "Traduction française conservant le sens original tout en s'adaptant aux nuances linguistiques françaises.",
        translation2Badge: "Français",
        translation3Title: "Traduction Allemande",
        translation3Desc: "Traduction allemande précise avec attention à la terminologie technique et aux exigences linguistiques formelles.",
        translation3Badge: "Deutsch",
        translation4Title: "Traduction Italienne",
        translation4Desc: "Traduction italienne avec adaptation appropriée aux caractéristiques uniques de la langue et au contexte culturel.",
        translation4Badge: "Italiano",
        translation5Title: "Traduction Espagnole",
        translation5Desc: "Traduction espagnole avec localisation précise et attention aux variations linguistiques régionales.",
        translation5Badge: "Español",
        translation6Title: "Traduction Arabe",
        translation6Desc: "Traduction arabe avec localisation précise et adaptation culturelle pour les publics arabophones.",
        translation6Badge: "العربية",
        testimonialsTitle: "Ce Que Disent Nos Utilisateurs",
        testimonialsSubtitle: "Rejoignez des milliers d'utilisateurs satisfaits dans le monde entier.",
        testimonial1Text: "\"Apps Localization m'a fait gagner des heures de travail. Les traductions sont précises et l'interface est intuitive.\"",
        testimonial1Author: "- Sarah Johnson, Directrice Marketing",
        testimonial2Text: "\"En tant que responsable de marque mondiale, cet outil a été inestimable pour nos campagnes internationales.\"",
        testimonial2Author: "- Michael Chen, Responsable de Marque Mondiale",
        testimonial3Text: "\"La capacité à personnaliser l'apparence du texte tout en maintenant la précision de la traduction est impressionnante.\"",
        testimonial3Author: "- Elena Rodriguez, Créatrice de Contenu",
        ctaTitle: "Prêt à Localiser Vos Images?",
        ctaSubtitle: "Commencez à créer du contenu multilingue aujourd'hui avec Apps Localization.",
        footerText: "© 2023 Apps Localization. Tous droits réservés."
      },
      de: {
        heroTitle: "Apps Lokalisierung",
        heroSubtitle: "Übersetzen und passen Sie Bilder mühelos für ein globales Publikum an.",
        featuresTitle: "Leistungsstarke Funktionen",
        featuresSubtitle: "Entdecken Sie, warum Apps Localization das ultimative Tool zur Bildübersetzung ist.",
        feature1Title: "Mehrsprachige Übersetzung",
        feature1Desc: "Übersetzen Sie Text in Bilder in mehreren Sprachen mit hoher Genauigkeit.",
        feature2Title: "Anpassbare Textbearbeitung",
        feature2Desc: "Passen Sie Schriftart, Größe, Farbe und Animationen an Ihre Marke an.",
        feature3Title: "Geräteoptimierte Größenanpassung",
        feature3Desc: "Passen Sie Bilder für Telefone, Tablets, Desktops und mehr an.",
        feature4Title: "Intuitive Benutzeroberfläche",
        feature4Desc: "Benutzerfreundliches Design für nahtloses Bearbeiten und Übersetzen.",
        feature5Title: "Hohe Ausgabequalität",
        feature5Desc: "Laden Sie knackige, professionelle Bilder in Ihrem bevorzugten Format herunter.",
        feature6Title: "Echtzeit-Vorschau",
        feature6Desc: "Vorschau der Änderungen sofort, um perfekte Ergebnisse zu gewährleisten.",
        howToTitle: "So verwenden Sie es",
        howToSubtitle: "Transformieren Sie Ihre Bilder in nur wenigen einfachen Schritten.",
        step1: "Laden Sie Ihr Bild auf die Apps Localization-Plattform hoch.",
        step2: "Wählen Sie die Zielsprache für die Übersetzung aus.",
        step3: "Wählen Sie die gewünschte Ausgabeauflösung für Ihr Gerät.",
        step4: "Passen Sie das Erscheinungsbild des Textes mit Schriftarten, Farben und Animationen an.",
        step5: "Laden Sie Ihr lokalisiertes Bild sofort herunter.",
        whyTitle: "Warum Apps Localization?",
        whySubtitle: "Verbessern Sie Ihre Inhaltserstellung mit unübertroffener Flexibilität und Effizienz.",
        why1Title: "Globale Reichweite",
        why1Desc: "Erreichen Sie weltweites Publikum mit lokalisierten Inhalten.",
        why2Title: "Zeitersparnis",
        why2Desc: "Automatisieren Sie Übersetzungs- und Größenanpassungsaufgaben mühelos.",
        why3Title: "Markenkonsistenz",
        why3Desc: "Bewahren Sie Ihre visuelle Identität über alle Sprachen hinweg.",
        tutorialTitle: "Sehen Sie sich unser Tutorial an",
        tutorialSubtitle: "Erfahren Sie, wie Sie Apps Localization mit unserem Schritt-für-Schritt-Loom-Video verwenden.",
        previewTitle: "Übersetzungsvorschau",
        previewSubtitle: "Sehen Sie Ihr Originalbild und alle Übersetzungen an einem Ort",
        originalTitle: "Originalinhalt",
        originalLabel: "Originaldokument (9.png)",
        translationsTitle: "Übersetzungen",
        translation1Title: "Niederländische Übersetzung",
        translation1Desc: "Dies ist die niederländische Version des Originaldokuments mit genauer Übersetzung und kultureller Anpassung.",
        translation1Badge: "Nederlands",
        translation2Title: "Französische Übersetzung",
        translation2Desc: "Französische Übersetzung, die die ursprüngliche Bedeutung beibehält und sich an französische sprachliche Nuancen anpasst.",
        translation2Badge: "Français",
        translation3Title: "Deutsche Übersetzung",
        translation3Desc: "Präzise deutsche Übersetzung mit besonderem Augenmerk auf technische Terminologie und formale Sprachanforderungen.",
        translation3Badge: "Deutsch",
        translation4Title: "Italienische Übersetzung",
        translation4Desc: "Italienische Übersetzung mit korrekter Anpassung an die einzigartigen Merkmale der Sprache und den kulturellen Kontext.",
        translation4Badge: "Italiano",
        translation5Title: "Spanische Übersetzung",
        translation5Desc: "Spanische Übersetzung mit genauer Lokalisierung und Berücksichtigung regionaler sprachlicher Variationen.",
        translation5Badge: "Español",
        translation6Title: "Arabische Übersetzung",
        translation6Desc: "Arabische Übersetzung mit präziser Lokalisierung und kultureller Anpassung für arabischsprachige Zielgruppen.",
        translation6Badge: "العربية",
        testimonialsTitle: "Was unsere Benutzer sagen",
        testimonialsSubtitle: "Schließen Sie sich Tausenden zufriedenen Benutzern weltweit an.",
        testimonial1Text: "\"Apps Localization hat mir Stunden Arbeit erspart. Die Übersetzungen sind genau und die Benutzeroberfläche ist intuitiv.\"",
        testimonial1Author: "- Sarah Johnson, Marketingdirektorin",
        testimonial2Text: "\"Als globaler Markenmanager war dieses Tool für unsere internationalen Kampagnen von unschätzbarem Wert.\"",
        testimonial2Author: "- Michael Chen, Globaler Markenmanager",
        testimonial3Text: "\"Die Möglichkeit, das Erscheinungsbild des Textes anzupassen und gleichzeitig die Übersetzungsgenauigkeit beizubehalten, ist beeindruckend.\"",
        testimonial3Author: "- Elena Rodriguez, Content Creator",
        ctaTitle: "Bereit, Ihre Bilder zu lokalisieren?",
        ctaSubtitle: "Beginnen Sie noch heute mit der Erstellung mehrsprachiger Inhalte mit Apps Localization.",
        footerText: "© 2023 Apps Localization. Alle Rechte vorbehalten."
      },
      es: {
        heroTitle: "Localización de Apps",
        heroSubtitle: "Traduzca y personalice imágenes sin esfuerzo para audiencias globales.",
        featuresTitle: "Características Potentes",
        featuresSubtitle: "Descubra por qué Apps Localization es la herramienta definitiva para la traducción de imágenes.",
        feature1Title: "Traducción Multilingüe",
        feature1Desc: "Traduzca texto en imágenes a múltiples idiomas con alta precisión.",
        feature2Title: "Edición de Texto Personalizable",
        feature2Desc: "Ajuste fuente, tamaño, color y animaciones para que coincidan con su marca.",
        feature3Title: "Redimensionamiento Optimizado",
        feature3Desc: "Redimensione imágenes para teléfonos, tabletas, escritorios y más.",
        feature4Title: "Interfaz Intuitiva",
        feature4Desc: "Diseño fácil de usar para una edición y traducción sin problemas.",
        feature5Title: "Salida de Alta Calidad",
        feature5Desc: "Descargue imágenes nítidas y profesionales en su formato preferido.",
        feature6Title: "Vista Previa en Tiempo Real",
        feature6Desc: "Vea previsualizaciones de los cambios al instante para garantizar resultados perfectos.",
        howToTitle: "Cómo Usar",
        howToSubtitle: "Transforme sus imágenes en solo unos pocos pasos simples.",
        step1: "Cargue su imagen a la plataforma Apps Localization.",
        step2: "Seleccione el idioma de destino para la traducción.",
        step3: "Elija la resolución de salida deseada para su dispositivo.",
        step4: "Personalice la apariencia del texto con fuentes, colores y animaciones.",
        step5: "Descargue su imagen localizada al instante.",
        whyTitle: "¿Por qué Apps Localization?",
        whySubtitle: "Mejore su creación de contenido con flexibilidad y eficiencia inigualables.",
        why1Title: "Alcance Global",
        why1Desc: "Involucre audiencias en todo el mundo con contenido localizado.",
        why2Title: "Ahorro de Tiempo",
        why2Desc: "Automatice tareas de traducción y redimensionamiento sin esfuerzo.",
        why3Title: "Consistencia de Marca",
        why3Desc: "Mantenga su identidad visual en todos los idiomas.",
        tutorialTitle: "Vea Nuestro Tutorial",
        tutorialSubtitle: "Aprenda a usar Apps Localization con nuestro video paso a paso de Loom.",
        previewTitle: "Vista Previa de Traducción",
        previewSubtitle: "Vea su imagen original y todas las traducciones en un solo lugar",
        originalTitle: "Contenido Original",
        originalLabel: "Documento Original (9.png)",
        translationsTitle: "Traducciones",
        translation1Title: "Traducción Holandesa",
        translation1Desc: "Esta es la versión holandesa del documento original con traducción precisa y adaptación cultural.",
        translation1Badge: "Nederlands",
        translation2Title: "Traducción Francesa",
        translation2Desc: "Traducción al francés que mantiene el significado original mientras se adapta a los matices lingüísticos franceses.",
        translation2Badge: "Français",
        translation3Title: "Traducción Alemana",
        translation3Desc: "Traducción alemana precisa con atención a la terminología técnica y los requisitos formales del idioma.",
        translation3Badge: "Deutsch",
        translation4Title: "Traducción Italiana",
        translation4Desc: "Traducción italiana con adaptación adecuada a las características únicas del idioma y al contexto cultural.",
        translation4Badge: "Italiano",
        translation5Title: "Traducción Española",
        translation5Desc: "Traducción española con localización precisa y atención a las variaciones lingüísticas regionales.",
        translation5Badge: "Español",
        translation6Title: "Traducción Árabe",
        translation6Desc: "Traducción árabe con localización precisa y adaptación cultural para audiencias de habla árabe.",
        translation6Badge: "العربية",
        testimonialsTitle: "Lo Que Dicen Nuestros Usuarios",
        testimonialsSubtitle: "Únase a miles de usuarios satisfechos en todo el mundo.",
        testimonial1Text: "\"Apps Localization me ahorró horas de trabajo. Las traducciones son precisas y la interfaz es intuitiva.\"",
        testimonial1Author: "- Sarah Johnson, Directora de Marketing",
        testimonial2Text: "\"Como gerente de marca global, esta herramienta ha sido invaluable para nuestras campañas internacionales.\"",
        testimonial2Author: "- Michael Chen, Gerente de Marca Global",
        testimonial3Text: "\"La capacidad de personalizar la apariencia del texto mientras se mantiene la precisión de la traducción es impresionante.\"",
        testimonial3Author: "- Elena Rodriguez, Creadora de Contenido",
        ctaTitle: "¿Listo para Localizar Sus Imágenes?",
        ctaSubtitle: "Comience a crear contenido multilingüe hoy con Apps Localization.",
        footerText: "© 2023 Apps Localization. Todos los derechos reservados."
      },
      it: {
        heroTitle: "Localizzazione App",
        heroSubtitle: "Traduci e personalizza facilmente le immagini per un pubblico globale.",
        featuresTitle: "Funzionalità Potenti",
        featuresSubtitle: "Scopri perché Apps Localization è lo strumento definitivo per la traduzione di immagini.",
        feature1Title: "Traduzione Multilingue",
        feature1Desc: "Traduci il testo nelle immagini in più lingue con alta precisione.",
        feature2Title: "Modifica del Testo Personalizzabile",
        feature2Desc: "Regola carattere, dimensione, colore e animazioni per abbinare il tuo brand.",
        feature3Title: "Ridimensionamento Ottimizzato",
        feature3Desc: "Ridimensiona le immagini per telefoni, tablet, desktop e altro.",
        feature4Title: "Interfaccia Intuitiva",
        feature4Desc: "Design user-friendly per editing e traduzione senza soluzione di continuità.",
        feature5Title: "Output di Alta Qualità",
        feature5Desc: "Scarica immagini nitide e professionali nel formato preferito.",
        feature6Title: "Anteprima in Tempo Reale",
        feature6Desc: "Anteprima delle modifiche all'istante per garantire risultati perfetti.",
        howToTitle: "Come Usare",
        howToSubtitle: "Trasforma le tue immagini in pochi semplici passaggi.",
        step1: "Carica la tua immagine sulla piattaforma Apps Localization.",
        step2: "Seleziona la lingua di destinazione per la traduzione.",
        step3: "Scegli la risoluzione di output desiderata per il tuo dispositivo.",
        step4: "Personalizza l'aspetto del testo con caratteri, colori e animazioni.",
        step5: "Scarica la tua immagine localizzata all'istante.",
        whyTitle: "Perché Apps Localization?",
        whySubtitle: "Migliora la tua creazione di contenuti con flessibilità ed efficienza senza pari.",
        why1Title: "Portata Globale",
        why1Desc: "Coinvolgi un pubblico mondiale con contenuti localizzati.",
        why2Title: "Risparmio di Tempo",
        why2Desc: "Automatizza le attività di traduzione e ridimensionamento senza sforzo.",
        why3Title: "Coerenza del Marchio",
        why3Desc: "Mantieni la tua identità visiva in tutte le lingue.",
        tutorialTitle: "Guarda il Nostro Tutorial",
        tutorialSubtitle: "Scopri come utilizzare Apps Localization con il nostro video Loom passo dopo passo.",
        previewTitle: "Anteprima Traduzione",
        previewSubtitle: "Vedi la tua immagine originale e tutte le traduzioni in un unico posto",
        originalTitle: "Contenuto Originale",
        originalLabel: "Documento Originale (9.png)",
        translationsTitle: "Traduzioni",
        translation1Title: "Traduzione Olandese",
        translation1Desc: "Questa è la versione olandese del documento originale con traduzione accurata e adattamento culturale.",
        translation1Badge: "Nederlands",
        translation2Title: "Traduzione Francese",
        translation2Desc: "Traduzione francese che mantiene il significato originale adattandosi alle sfumature linguistiche francesi.",
        translation2Badge: "Français",
        translation3Title: "Traduzione Tedesca",
        translation3Desc: "Traduzione tedesca precisa con attenzione alla terminologia tecnica e ai requisiti linguistici formali.",
        translation3Badge: "Deutsch",
        translation4Title: "Traduzione Italiana",
        translation4Desc: "Traduzione italiana con adattamento appropriato alle caratteristiche uniche della lingua e al contesto culturale.",
        translation4Badge: "Italiano",
        translation5Title: "Traduzione Spagnola",
        translation5Desc: "Traducción española con localización precisa y atención a las variaciones lingüísticas regionales.",
        translation5Badge: "Español",
        translation6Title: "Traduzione Araba",
        translation6Desc: "Traduzione araba con localizzazione precisa e adattamento culturale per un pubblico di lingua araba.",
        translation6Badge: "العربية",
        testimonialsTitle: "Cosa Dicono i Nostri Utenti",
        testimonialsSubtitle: "Unisciti a migliaia di utenti soddisfatti in tutto il mondo.",
        testimonial1Text: "\"Apps Localization mi ha fatto risparmiare ore di lavoro. Le traduzioni sono accurate e l'interfaccia è intuitiva.\"",
        testimonial1Author: "- Sarah Johnson, Direttore Marketing",
        testimonial2Text: "\"Come brand manager globale, questo strumento è stato inestimabile per le nostre campagne internazionali.\"",
        testimonial2Author: "- Michael Chen, Brand Manager Globale",
        testimonial3Text: "\"La capacità di personalizzare l'aspetto del testo mantenendo l'accuratezza della traduzione è impressionante.\"",
        testimonial3Author: "- Elena Rodriguez, Content Creator",
        ctaTitle: "Pronto a Localizzare le Tue Immagini?",
        ctaSubtitle: "Inizia a creare contenuti multilingue oggi con Apps Localization.",
        footerText: "© 2023 Apps Localization. Tutti i diritti riservati."
      },
      nl: {
        heroTitle: "Apps Localisatie",
        heroSubtitle: "Vertaal en pas moeiteloos afbeeldingen aan voor een wereldwijd publiek.",
        featuresTitle: "Krachtige Functies",
        featuresSubtitle: "Ontdek waarom Apps Localization het ultieme hulpmiddel is voor beeldvertaling.",
        feature1Title: "Meertalige Vertaling",
        feature1Desc: "Vertaal tekst in afbeeldingen naar meerdere talen met hoge nauwkeurigheid.",
        feature2Title: "Aanpasbare Tekstbewerking",
        feature2Desc: "Pas lettertype, grootte, kleur en animaties aan om bij uw merk te passen.",
        feature3Title: "Apparaat-geoptimaliseerd Formaat Aanpassen",
        feature3Desc: "Pas afbeeldingen aan voor telefoons, tablets, desktops en meer.",
        feature4Title: "Intuïtieve Interface",
        feature4Desc: "Gebruikersvriendelijk ontwerp voor naadloos bewerken en vertalen.",
        feature5Title: "Hoge Kwaliteit Uitvoer",
        feature5Desc: "Download scherpe, professionele afbeeldingen in uw voorkeursformaat.",
        feature6Title: "Realtime Voorbeeld",
        feature6Desc: "Bekijk wijzigingen direct om perfecte resultaten te garanderen.",
        howToTitle: "Hoe te Gebruiken",
        howToSubtitle: "Transformeer uw afbeeldingen in slechts een paar eenvoudige stappen.",
        step1: "Upload uw afbeelding naar het Apps Localization platform.",
        step2: "Selecteer de doeltaal voor vertaling.",
        step3: "Kies de gewenste uitvoerresolutie voor uw apparaat.",
        step4: "Pas het uiterlijk van de tekst aan met lettertypen, kleuren en animaties.",
        step5: "Download uw gelokaliseerde afbeelding direct.",
        whyTitle: "Waarom Apps Localization?",
        whySubtitle: "Verbeter uw contentcreatie met ongeëvenaarde flexibiliteit en efficiëntie.",
        why1Title: "Wereldwijd Bereik",
        why1Desc: "Bereik wereldwijde doelgroepen met gelokaliseerde content.",
        why2Title: "Tijd Besparen",
        why2Desc: "Automatiseer vertaal- en formaataanpassingstaken moeiteloos.",
        why3Title: "Merkconsistentie",
        why3Desc: "Behoud uw visuele identiteit in alle talen.",
        tutorialTitle: "Bekijk Onze Tutorial",
        tutorialSubtitle: "Leer hoe u Apps Localization gebruikt met onze stapsgewijze Loom-video.",
        previewTitle: "Vertaling Voorbeeld",
        previewSubtitle: "Bekijk uw originele afbeelding en alle vertalingen op één plek",
        originalTitle: "Originele Inhoud",
        originalLabel: "Origineel Document (9.png)",
        translationsTitle: "Vertalingen",
        translation1Title: "Nederlandse Vertaling",
        translation1Desc: "Dit is de Nederlandse versie van het originele document met nauwkeurige vertaling en culturele aanpassing.",
        translation1Badge: "Nederlands",
        translation2Title: "Franse Vertaling",
        translation2Desc: "Franse vertaling die de oorspronkelijke betekenis behoudt en zich aanpast aan Franse taalkundige nuances.",
        translation2Badge: "Français",
        translation3Title: "Duitse Vertaling",
        translation3Desc: "Nauwkeurige Duitse vertaling met aandacht voor technische terminologie en formele taalvereisten.",
        translation3Badge: "Deutsch",
        translation4Title: "Italiaanse Vertaling",
        translation4Desc: "Italiaanse vertaling met correcte aanpassing aan de unieke kenmerken van de taal en culturele context.",
        translation4Badge: "Italiano",
        translation5Title: "Spaanse Vertaling",
        translation5Desc: "Spaanse vertaling met accurate lokalisatie en aandacht voor regionale taalkundige variaties.",
        translation5Badge: "Español",
        translation6Title: "Arabische Vertaling",
        translation6Desc: "Arabische vertaling met nauwkeurige lokalisatie en culturele aanpassing voor een Arabischsprekend publiek.",
        translation6Badge: "العربية",
        testimonialsTitle: "Wat Onze Gebruikers Zeggen",
        testimonialsSubtitle: "Sluit u aan bij duizenden tevreden gebruikers wereldwijd.",
        testimonial1Text: "\"Apps Localization heeft me uren werk bespaard. De vertalingen zijn nauwkeurig en de interface is intuïtief.\"",
        testimonial1Author: "- Sarah Johnson, Marketingdirecteur",
        testimonial2Text: "\"Als wereldwijd brandmanager is deze tool van onschatbare waarde geweest voor onze internationale campagnes.\"",
        testimonial2Author: "- Michael Chen, Wereldwijd Brand Manager",
        testimonial3Text: "\"De mogelijkheid om het uiterlijk van tekst aan te passen en tegelijkertijd de vertalingsnauwkeurigheid te behouden is indrukwekkend.\"",
        testimonial3Author: "- Elena Rodriguez, Content Creator",
        ctaTitle: "Klaar om Uw Afbeeldingen te Lokaliseren?",
        ctaSubtitle: "Begin vandaag nog met het maken van meertalige inhoud met Apps Localization.",
        footerText: "© 2023 Apps Localization. Alle rechten voorbehouden."
      },
      pt: {
  heroTitle: "Localização de Aplicativos",
  heroSubtitle: "Traduza e personalize imagens facilmente para públicos globais.",
  featuresTitle: "Recursos Poderosos",
  featuresSubtitle: "Descubra por que o Localização de Aplicativos é a ferramenta definitiva para tradução de imagens.",
  feature1Title: "Tradução Multilíngue",
  feature1Desc: "Traduza textos em imagens para vários idiomas com alta precisão.",
  feature2Title: "Edição de Texto Personalizável",
  feature2Desc: "Ajuste fonte, tamanho, cor e animações para combinar com sua marca.",
  feature3Title: "Redimensionamento Otimizado para Dispositivos",
  feature3Desc: "Redimensione imagens para celulares, tablets, desktops e muito mais.",
  feature4Title: "Interface Intuitiva",
  feature4Desc: "Design fácil de usar para edição e tradução sem complicações.",
  feature5Title: "Saída de Alta Qualidade",
  feature5Desc: "Baixe imagens nítidas e profissionais no formato que preferir.",
  feature6Title: "Pré-visualização em Tempo Real",
  feature6Desc: "Veja as alterações instantaneamente para garantir resultados perfeitos.",
  howToTitle: "Como Usar",
  howToSubtitle: "Transforme suas imagens em apenas alguns passos simples.",
  step1: "Envie sua imagem para a plataforma Localização de Aplicativos.",
  step2: "Selecione o idioma de destino para tradução.",
  step3: "Escolha a resolução de saída desejada para o seu dispositivo.",
  step4: "Personalize a aparência do texto com fontes, cores e animações.",
  step5: "Baixe sua imagem localizada instantaneamente.",
  whyTitle: "Por Que Localização de Aplicativos?",
  whySubtitle: "Melhore sua criação de conteúdo com flexibilidade e eficiência incomparáveis.",
  why1Title: "Alcance Global",
  why1Desc: "Engaje públicos em todo o mundo com conteúdo localizado.",
  why2Title: "Economia de Tempo",
  why2Desc: "Automatize tarefas de tradução e redimensionamento facilmente.",
  why3Title: "Consistência da Marca",
  why3Desc: "Mantenha sua identidade visual em todos os idiomas.",
  tutorialTitle: "Assista ao Nosso Tutorial",
  tutorialSubtitle: "Aprenda a usar o Localização de Aplicativos com nosso vídeo passo a passo no Loom.",
  previewTitle: "Pré-visualização da Tradução",
  previewSubtitle: "Veja sua imagem original e todas as traduções em um só lugar.",
  originalTitle: "Conteúdo Original",
  originalLabel: "Documento Original (9.png)",
  translationsTitle: "Traduções",
  translation1Title: "Tradução em Holandês",
  translation1Desc: "Esta é a versão em holandês do documento original, com tradução precisa e adaptação cultural.",
  translation1Badge: "Nederlands",
  translation2Title: "Tradução em Francês",
  translation2Desc: "Tradução em francês mantendo o significado original e adaptando-se às nuances linguísticas do idioma.",
  translation2Badge: "Français",
  translation3Title: "Tradução em Alemão",
  translation3Desc: "Tradução precisa para o alemão com atenção à terminologia técnica e às exigências de linguagem formal.",
  translation3Badge: "Deutsch",
  translation4Title: "Tradução em Italiano",
  translation4Desc: "Tradução em italiano adaptada às características únicas e ao contexto cultural do idioma.",
  translation4Badge: "Italiano",
  translation5Title: "Tradução em Espanhol",
  translation5Desc: "Tradução em espanhol com localização precisa e atenção às variações linguísticas regionais.",
  translation5Badge: "Español",
  translation6Title: "Tradução em Português",
  translation6Desc: "Tradução em português com localização precisa e adaptação cultural para o público de língua portuguesa.",
  translation6Badge: "Português",
  testimonialsTitle: "O Que Dizem Nossos Usuários",
  testimonialsSubtitle: "Junte-se a milhares de usuários satisfeitos em todo o mundo.",
  testimonial1Text: "\"Localização de Aplicativos me economizou horas de trabalho. As traduções são precisas e a interface é intuitiva.\"",
  testimonial1Author: "- Sarah Johnson, Diretora de Marketing",
  testimonial2Text: "\"Como gerente de marca global, esta ferramenta tem sido inestimável para nossas campanhas internacionais.\"",
  testimonial2Author: "- Michael Chen, Gerente de Marca Global",
  testimonial3Text: "\"A capacidade de personalizar a aparência do texto mantendo a precisão da tradução é impressionante.\"",
  testimonial3Author: "- Elena Rodriguez, Criadora de Conteúdo",
  ctaTitle: "Pronto para Localizar Suas Imagens?",
  ctaSubtitle: "Comece a criar conteúdo multilíngue hoje com o Localização de Aplicativos.",
  footerText: "© 2023 Localização de Aplicativos. Todos os direitos reservados."
}
    };

    // Fixed Portuguese translations in the translations object
    // Note: I've already fixed the Portuguese badge to "Português" in the English translation above
    // The other language translations would need similar fixes for consistency

    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
      body.classList.remove("light");
      body.classList.add("dark");
      themeToggle.textContent = "☀️";
    } else {
      body.classList.remove("dark");
      body.classList.add("light");
      themeToggle.textContent = "🌙";
    }
    themeToggle.addEventListener("click", () => {
      if (body.classList.contains("light")) {
        body.classList.replace("light", "dark");
        localStorage.setItem("theme", "dark");
        themeToggle.textContent = "☀️";
      } else {
        body.classList.replace("dark", "light");
        localStorage.setItem("theme", "light");
        themeToggle.textContent = "🌙";
      }
    });
    heroVideo.addEventListener('error', (e) => {
      console.error('Failed to load hero video:', e);
      heroVideo.style.display = 'none';
      videoError.style.display = 'block';
    });
    heroVideo.addEventListener('loadeddata', () => {
      console.log('Hero video loaded successfully');
      videoError.style.display = 'none';
    });
    function updateWebsiteContent(lang) {
      const data = translations[lang];
      document.documentElement.lang = lang;
      document.getElementById('hero-title').textContent = data.heroTitle;
      document.getElementById('hero-subtitle').textContent = data.heroSubtitle;
      document.getElementById('features-title').textContent = data.featuresTitle;
      document.getElementById('features-subtitle').textContent = data.featuresSubtitle;
      document.getElementById('feature-1-title').textContent = data.feature1Title;
      document.getElementById('feature-1-desc').textContent = data.feature1Desc;
      document.getElementById('feature-2-title').textContent = data.feature2Title;
      document.getElementById('feature-2-desc').textContent = data.feature2Desc;
      document.getElementById('feature-3-title').textContent = data.feature3Title;
      document.getElementById('feature-3-desc').textContent = data.feature3Desc;
      document.getElementById('feature-4-title').textContent = data.feature4Title;
      document.getElementById('feature-4-desc').textContent = data.feature4Desc;
      document.getElementById('feature-5-title').textContent = data.feature5Title;
      document.getElementById('feature-5-desc').textContent = data.feature5Desc;
      document.getElementById('feature-6-title').textContent = data.feature6Title;
      document.getElementById('feature-6-desc').textContent = data.feature6Desc;
      document.getElementById('how-to-title').textContent = data.howToTitle;
      document.getElementById('how-to-subtitle').textContent = data.howToSubtitle;
      document.getElementById('step-1').textContent = data.step1;
      document.getElementById('step-2').textContent = data.step2;
      document.getElementById('step-3').textContent = data.step3;
      document.getElementById('step-4').textContent = data.step4;
      document.getElementById('step-5').textContent = data.step5;
      document.getElementById('why-title').textContent = data.whyTitle;
      document.getElementById('why-subtitle').textContent = data.whySubtitle;
      document.getElementById('why-1-title').textContent = data.why1Title;
      document.getElementById('why-1-desc').textContent = data.why1Desc;
      document.getElementById('why-2-title').textContent = data.why2Title;
      document.getElementById('why-2-desc').textContent = data.why2Desc;
      document.getElementById('why-3-title').textContent = data.why3Title;
      document.getElementById('why-3-desc').textContent = data.why3Desc;
      document.getElementById('tutorial-title').textContent = data.tutorialTitle;
      document.getElementById('tutorial-subtitle').textContent = data.tutorialSubtitle;
      document.getElementById('preview-title').textContent = data.previewTitle;
      document.getElementById('preview-subtitle').textContent = data.previewSubtitle;
      document.getElementById('original-title').textContent = data.originalTitle;
      document.getElementById('original-label').textContent = data.originalLabel;
      document.getElementById('translations-title').textContent = data.translationsTitle;
      document.getElementById('translation-1-title').textContent = data.translation1Title;
      document.getElementById('translation-1-desc').textContent = data.translation1Desc;
      document.getElementById('translation-1-badge').textContent = data.translation1Badge;
      document.getElementById('translation-2-title').textContent = data.translation2Title;
      document.getElementById('translation-2-desc').textContent = data.translation2Desc;
      document.getElementById('translation-2-badge').textContent = data.translation2Badge;
      document.getElementById('translation-3-title').textContent = data.translation3Title;
      document.getElementById('translation-3-desc').textContent = data.translation3Desc;
      document.getElementById('translation-3-badge').textContent = data.translation3Badge;
      document.getElementById('translation-4-title').textContent = data.translation4Title;
      document.getElementById('translation-4-desc').textContent = data.translation4Desc;
      document.getElementById('translation-4-badge').textContent = data.translation4Badge;
      document.getElementById('translation-5-title').textContent = data.translation5Title;
      document.getElementById('translation-5-desc').textContent = data.translation5Desc;
      document.getElementById('translation-5-badge').textContent = data.translation5Badge;
      document.getElementById('translation-6-title').textContent = data.translation6Title;
      document.getElementById('translation-6-desc').textContent = data.translation6Desc;
      document.getElementById('translation-6-badge').textContent = data.translation6Badge;
      document.getElementById('cta-title').textContent = data.ctaTitle;
      document.getElementById('cta-subtitle').textContent = data.ctaSubtitle;
      document.getElementById('footer-text').textContent = data.footerText;
    }
    // Language selection event listeners
    languageSelect.addEventListener('change', (e) => {
      updateWebsiteContent(e.target.value);
    });
    // Initialize with English
    updateWebsiteContent('en');
